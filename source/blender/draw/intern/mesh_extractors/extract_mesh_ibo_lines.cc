/* SPDX-FileCopyrightText: 2021 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup draw
 */

#include "BLI_array_utils.hh"

#include "GPU_index_buffer.hh"

#include "extract_mesh.hh"

#include "draw_cache_inline.hh"
#include "draw_subdivision.hh"

namespace blender::draw {

static IndexMask calc_mesh_edge_visibility(const MeshRenderData &mr,
                                           const IndexMask &mask,
                                           IndexMaskMemory &memory)
{
  IndexMask visible = mask;
  if (!mr.mesh->runtime->subsurf_optimal_display_edges.is_empty()) {
    const BoundedBitSpan visible_bits = mr.mesh->runtime->subsurf_optimal_display_edges;
    visible = IndexMask::from_bits(visible, visible_bits, memory);
  }
  if (!mr.hide_edge.is_empty()) {
    visible = IndexMask::from_bools_inverse(visible, mr.hide_edge, memory);
  }
  if (mr.hide_unmapped_edges && mr.orig_index_edge != nullptr) {
    const int *orig_index = mr.orig_index_edge;
    visible = IndexMask::from_predicate(visible, GrainSize(4096), memory, [&](const int64_t i) {
      return orig_index[i] != ORIGINDEX_NONE;
    });
  }
  return visible;
}

static void fill_loose_lines_ibo(const MeshRenderData &mr,
                                 const IndexMask &visible,
                                 MutableSpan<uint2> data)
{
  /* Vertices for loose edges are not shared in the GPU vertex buffers, so the indices are simply
   * an increasing contiguous range. Ideally this would be generated on the GPU itself, or just
   * unnecessary, but a large number of loose edges isn't expected to be a common performance
   * bottleneck either. */
  threading::memory_bandwidth_bound_task(data.size_in_bytes(), [&]() {
    const uint loose_start = mr.corners_num;
    if (visible.size() == mr.loose_edges.size()) {
      array_utils::fill_index_range(data.cast<uint>(), loose_start);
    }
    else {
      visible.foreach_index_optimized<int>(GrainSize(4096), [&](const int i, const int pos) {
        data[pos] = loose_start + uint2(i * 2 + 0, i * 2 + 1);
      });
    }
  });
}

static IndexMask calc_visible_loose_edge_indices(const MeshRenderData &mr, IndexMaskMemory &memory)
{
  const Span<int> loose_edges = mr.loose_edges;
  IndexMask visible(loose_edges.size());
  if (!mr.hide_edge.is_empty()) {
    const Span<bool> hide_edge = mr.hide_edge;
    visible = IndexMask::from_predicate(
        visible, GrainSize(4096), memory, [&](const int i) { return !hide_edge[loose_edges[i]]; });
  }
  if (mr.hide_unmapped_edges && mr.orig_index_edge != nullptr) {
    const int *orig_index = mr.orig_index_edge;
    visible = IndexMask::from_predicate(visible, GrainSize(4096), memory, [&](const int64_t i) {
      return orig_index[loose_edges[i]] != ORIGINDEX_NONE;
    });
  }
  return visible;
}

static void extract_lines_mesh(const MeshRenderData &mr,
                               gpu::IndexBufPtr *lines,
                               gpu::IndexBufPtr *lines_loose,
                               bool &no_loose_wire)
{
  IndexMaskMemory memory;
  const IndexMask visible_loose_edges = calc_visible_loose_edge_indices(mr, memory);
  const int max_index = mr.corners_num + mr.loose_edges.size() * 2;

  no_loose_wire = visible_loose_edges.is_empty();

  if (lines_loose && !lines) {
    GPUIndexBufBuilder builder;
    GPU_indexbuf_init(&builder, GPU_PRIM_LINES, visible_loose_edges.size(), max_index);
    MutableSpan<uint2> data = GPU_indexbuf_get_data(&builder).cast<uint2>();
    fill_loose_lines_ibo(mr, visible_loose_edges, data);
    *lines_loose = gpu::IndexBufPtr(GPU_indexbuf_build_ex(&builder, 0, max_index, false));
    return;
  }

  const IndexMask all_loose_edges = IndexMask::from_indices(mr.loose_edges, memory);
  const IndexMask non_loose_edges = all_loose_edges.complement(IndexRange(mr.edges_num), memory);
  const IndexMask visible_non_loose_edges = calc_mesh_edge_visibility(mr, non_loose_edges, memory);

  GPUIndexBufBuilder builder;
  GPU_indexbuf_init(&builder,
                    GPU_PRIM_LINES,
                    visible_non_loose_edges.size() + visible_loose_edges.size(),
                    max_index);
  MutableSpan<uint2> data = GPU_indexbuf_get_data(&builder).cast<uint2>();

  /* This code fills the index buffer in a non-deterministic way, with non-atomic access to `used`.
   * This is okay because any of the possible face corner indices are correct, since they all
   * correspond to the same #Mesh vertex. `used` exists here as a performance optimization to
   * avoid writing to the VBO. */
  const OffsetIndices faces = mr.faces;
  const Span<int> corner_edges = mr.corner_edges;
  if (visible_non_loose_edges.size() == mr.edges_num) {
    /* All edges in the mesh are visible. The edges in the GPU buffer will have the same order as
     * the mesh's edges, so any remapping is unnecessary. Use a boolean array to avoid writing to
     * the same indices again multiple times from different threads. This is slightly beneficial
     * because booleans are 8 times smaller than the `uint2` for each edge. */
    Array<bool> used(mr.edges_num, false);
    threading::memory_bandwidth_bound_task(
        used.as_span().size_in_bytes() + data.size_in_bytes() + corner_edges.size_in_bytes(),
        [&]() {
          threading::parallel_for(faces.index_range(), 2048, [&](const IndexRange range) {
            for (const int face_index : range) {
              const IndexRange face = faces[face_index];
              for (const int corner : face) {
                const int edge = corner_edges[corner];
                if (!used[edge]) {
                  used[edge] = true;
                  data[edge] = edge_from_corners(face, corner);
                }
              }
            }
          });
        });
  }
  else {
    Array<int> map(mr.edges_num, -1);
    threading::memory_bandwidth_bound_task(
        map.as_span().size_in_bytes() + data.size_in_bytes() + corner_edges.size_in_bytes(),
        [&]() {
          index_mask::build_reverse_map(visible_non_loose_edges, map.as_mutable_span());
          threading::parallel_for(faces.index_range(), 2048, [&](const IndexRange range) {
            for (const int face_index : range) {
              const IndexRange face = faces[face_index];
              for (const int corner : face) {
                const int edge = corner_edges[corner];
                const int vbo_index = map[edge];
                if (vbo_index != -1) {
                  map[edge] = -1;
                  data[vbo_index] = edge_from_corners(face, corner);
                }
              }
            }
          });
        });
  }

  fill_loose_lines_ibo(mr, visible_loose_edges, data.take_back(visible_loose_edges.size()));

  *lines = gpu::IndexBufPtr(GPU_indexbuf_build_ex(&builder, 0, max_index, false));
  if (lines_loose) {
    *lines_loose = gpu::IndexBufPtr(GPU_indexbuf_create_subrange(
        lines->get(), visible_non_loose_edges.size() * 2, visible_loose_edges.size() * 2));
  }
}

static void extract_lines_bm(const MeshRenderData &mr,
                             gpu::IndexBufPtr *lines,
                             gpu::IndexBufPtr *lines_loose,
                             bool &no_loose_wire)
{
  const BMesh &bm = *mr.bm;
  const Span<int> loose_edges = mr.loose_edges;

  IndexMaskMemory memory;
  const IndexMask visible_loose_edges = IndexMask::from_predicate(
      loose_edges.index_range(), GrainSize(2048), memory, [&](const int i) {
        const BMEdge &edge = *BM_edge_at_index(&const_cast<BMesh &>(bm), loose_edges[i]);
        return !BM_elem_flag_test_bool(&edge, BM_ELEM_HIDDEN);
      });
  const int max_index = mr.corners_num + loose_edges.size() * 2;

  no_loose_wire = visible_loose_edges.is_empty();

  if (lines_loose && !lines) {
    GPUIndexBufBuilder builder;
    GPU_indexbuf_init(&builder, GPU_PRIM_LINES, visible_loose_edges.size(), max_index);
    MutableSpan<uint2> data = GPU_indexbuf_get_data(&builder).cast<uint2>();
    fill_loose_lines_ibo(mr, visible_loose_edges, data);
    *lines_loose = gpu::IndexBufPtr(GPU_indexbuf_build_ex(&builder, 0, max_index, false));
    return;
  }

  const IndexMask all_loose_edges = IndexMask::from_indices(mr.loose_edges, memory);
  const IndexMask non_loose_edges = all_loose_edges.complement(IndexRange(bm.totedge), memory);
  const IndexMask visible_non_loose_edges = IndexMask::from_predicate(
      non_loose_edges, GrainSize(2048), memory, [&](const int i) {
        const BMEdge &edge = *BM_edge_at_index(&const_cast<BMesh &>(bm), i);
        return !BM_elem_flag_test_bool(&edge, BM_ELEM_HIDDEN);
      });

  GPUIndexBufBuilder builder;
  GPU_indexbuf_init(&builder,
                    GPU_PRIM_LINES,
                    visible_non_loose_edges.size() + visible_loose_edges.size(),
                    max_index);
  MutableSpan<uint2> data = GPU_indexbuf_get_data(&builder).cast<uint2>();

  /* Make use of BMesh's edge to loop topology knowledge to iterate over edges instead of
   * iterating over faces and defining edges implicitly as done in the #Mesh extraction. */
  visible_non_loose_edges.foreach_index(GrainSize(4096), [&](const int i, const int pos) {
    const BMEdge &edge = *BM_edge_at_index(&const_cast<BMesh &>(bm), i);
    data[pos] = uint2(BM_elem_index_get(edge.l), BM_elem_index_get(edge.l->next));
  });

  fill_loose_lines_ibo(mr, visible_loose_edges, data.take_back(visible_loose_edges.size()));

  *lines = gpu::IndexBufPtr(GPU_indexbuf_build_ex(&builder, 0, max_index, false));
  if (lines_loose) {
    *lines_loose = gpu::IndexBufPtr(GPU_indexbuf_create_subrange(
        lines->get(), visible_non_loose_edges.size() * 2, visible_loose_edges.size() * 2));
  }
}

void extract_lines(const MeshRenderData &mr,
                   gpu::IndexBufPtr *lines,
                   gpu::IndexBufPtr *lines_loose,
                   bool &no_loose_wire)
{
  if (mr.extract_type == MeshExtractType::Mesh) {
    extract_lines_mesh(mr, lines, lines_loose, no_loose_wire);
  }
  else {
    extract_lines_bm(mr, lines, lines_loose, no_loose_wire);
  }
}

static void extract_lines_loose_geom_subdiv(const DRWSubdivCache &subdiv_cache,
                                            const MeshRenderData &mr,
                                            const int edge_loose_offset,
                                            gpu::IndexBuf *ibo)
{
  const Span<int> loose_edges = mr.loose_edges;
  if (loose_edges.is_empty()) {
    return;
  }
  const int edges_per_edge = subdiv_edges_per_coarse_edge(subdiv_cache);
  const int loose_edges_num = subdiv_loose_edges_num(mr, subdiv_cache);

  /* Update flags for loose edges, points are already handled. */
  static const GPUVertFormat format = GPU_vertformat_from_attribute("data",
                                                                    gpu::VertAttrType::UINT_32);

  gpu::VertBuf *flags = GPU_vertbuf_calloc();
  GPU_vertbuf_init_with_format(*flags, format);

  GPU_vertbuf_data_alloc(*flags, loose_edges_num);

  MutableSpan<uint> flags_data = flags->data<uint>();

  switch (mr.extract_type) {
    case MeshExtractType::Mesh: {
      const int *orig_index_edge = (mr.hide_unmapped_edges) ? mr.orig_index_edge : nullptr;
      if (orig_index_edge == nullptr) {
        const Span<bool> hide_edge = mr.hide_edge;
        if (!hide_edge.is_empty()) {
          for (const int i : loose_edges.index_range()) {
            const bool value = hide_edge[loose_edges[i]];
            flags_data.slice(i * edges_per_edge, edges_per_edge).fill(value);
          }
        }
        else {
          flags_data.fill(0);
        }
      }
      else {
        if (mr.bm) {
          for (const int i : loose_edges.index_range()) {
            const BMEdge *bm_edge = bm_original_edge_get(mr, loose_edges[i]);
            const int value = (bm_edge) ? BM_elem_flag_test_bool(bm_edge, BM_ELEM_HIDDEN) : true;
            flags_data.slice(i * edges_per_edge, edges_per_edge).fill(value);
          }
        }
        else {
          const Span<bool> hide_edge = mr.hide_edge;
          if (!hide_edge.is_empty()) {
            for (const int i : loose_edges.index_range()) {
              const bool value = (orig_index_edge[loose_edges[i]] == ORIGINDEX_NONE) ?
                                     false :
                                     hide_edge[loose_edges[i]];
              flags_data.slice(i * edges_per_edge, edges_per_edge).fill(value);
            }
          }
          else {
            flags_data.fill(0);
          }
        }
      }
      break;
    }
    case MeshExtractType::BMesh: {
      BMesh *bm = mr.bm;
      for (const int i : loose_edges.index_range()) {
        const BMEdge *bm_edge = BM_edge_at_index(bm, loose_edges[i]);
        const bool value = BM_elem_flag_test_bool(bm_edge, BM_ELEM_HIDDEN);
        flags_data.slice(i * edges_per_edge, edges_per_edge).fill(value);
      }
      break;
    }
  }

  draw_subdiv_build_lines_loose_buffer(
      subdiv_cache, ibo, flags, uint(edge_loose_offset), loose_edges_num);

  GPU_vertbuf_discard(flags);
}

void extract_lines_subdiv(const DRWSubdivCache &subdiv_cache,
                          const MeshRenderData &mr,
                          gpu::IndexBufPtr *lines,
                          gpu::IndexBufPtr *lines_loose,
                          bool &no_loose_wire)
{
  const int loose_ibo_size = subdiv_loose_edges_num(mr, subdiv_cache) * 2;
  no_loose_wire = loose_ibo_size == 0;

  if (lines_loose && !lines) {
    *lines_loose = gpu::IndexBufPtr(GPU_indexbuf_build_on_device(loose_ibo_size));
    extract_lines_loose_geom_subdiv(subdiv_cache, mr, 0, lines_loose->get());
    return;
  }

  const int non_loose_ibo_size = subdiv_cache.num_subdiv_loops * 2;

  *lines = gpu::IndexBufPtr(GPU_indexbuf_build_on_device(non_loose_ibo_size + loose_ibo_size));
  if (non_loose_ibo_size > 0) {
    draw_subdiv_build_lines_buffer(subdiv_cache, lines->get());
  }
  extract_lines_loose_geom_subdiv(subdiv_cache, mr, non_loose_ibo_size, lines->get());

  if (lines_loose) {
    /* Multiply by 2 because these are edges indices. */
    *lines_loose = gpu::IndexBufPtr(
        GPU_indexbuf_create_subrange(lines->get(), non_loose_ibo_size, loose_ibo_size));
  }
}

}  // namespace blender::draw
