/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/memory.h"

#include "scene/colorspace.h"
#include "scene/image_cache.h"

#include "util/set.h"
#include "util/string.h"
#include "util/texture.h"
#include "util/thread.h"
#include "util/transform.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;
class ImageHandle;
class ImageKey;
class ImageMetaData;
class ImageManager;
class Progress;
class RenderStats;
class Scene;
class SceneParams;
class ColorSpaceProcessor;
class VDBImageLoader;

/* Image Parameters */
class ImageParams {
 public:
  bool animated = false;
  InterpolationType interpolation = INTERPOLATION_LINEAR;
  ExtensionType extension = EXTENSION_CLIP;
  ImageAlphaType alpha_type = IMAGE_ALPHA_AUTO;
  ustring colorspace;
  float frame = 0.0f;

  ImageParams() : colorspace(u_colorspace_raw) {}

  bool operator==(const ImageParams &other) const
  {
    return (animated == other.animated && interpolation == other.interpolation &&
            extension == other.extension && alpha_type == other.alpha_type &&
            colorspace == other.colorspace && frame == other.frame);
  }
};

/* Image MetaData
 *
 * Information about the image that is available before the image pixels are loaded. */
class ImageMetaData {
 public:
  /* Set by ImageLoader.load_metadata(). */
  int channels = 0;
  int64_t width = 0, height = 0, depth = 0;
  int64_t byte_size = 0;
  ImageDataType type = IMAGE_DATA_NUM_TYPES;

  /* Optional color space, defaults to raw. */
  ustring colorspace = u_colorspace_raw;
  string colorspace_file_hint;
  const char *colorspace_file_format = "";

  /* Optional transform for 3D images. */
  bool use_transform_3d = false;
  Transform transform_3d = transform_identity();

  /* Automatically set. */
  bool compress_as_srgb = false;
  bool associate_alpha = false;

  /* Tiling */
  uint32_t tile_size = 0;
  float4 average_color = zero_float4();

  ImageMetaData();
  bool operator==(const ImageMetaData &other) const;
  bool is_float() const;
  void finalize(const ImageAlphaType alpha_type);
};

/* Image loader base class, that can be subclassed to load image data
 * from custom sources (file, memory, procedurally generated, etc). */
class ImageLoader {
 public:
  ImageLoader();
  virtual ~ImageLoader() = default;

  /* Enable use of the texture cache for this image, if supported by the image loader. */
  virtual bool resolve_texture_cache(const bool /*auto_generate*/,
                                     const string & /*texture_cache_path*/,
                                     const ImageAlphaType /*alpha_type*/)
  {
    return false;
  }

  /* Load metadata without actual image yet, should be fast. */
  virtual bool load_metadata(ImageMetaData &metadata) = 0;

  /* Load full image pixels. */
  virtual bool load_pixels_full(const ImageMetaData &metadata, uint8_t *pixels) = 0;

  /* Load pixels for a single tile, if ImageMetaData.tile_size is set. */
  virtual bool load_pixels_tile(const ImageMetaData & /*metadata*/,
                                const int /*miplevel*/,
                                const int64_t /*x*/,
                                const int64_t /*y*/,
                                const int64_t /*w*/,
                                const int64_t /*h*/,
                                const int64_t /*x_stride*/,
                                const int64_t /*y_stride*/,
                                const int64_t /*padding*/,
                                const ExtensionType /*extension*/,
                                uint8_t * /*pixels*/)
  {
    return false;
  }

  /* TODO: Drop file handle to avoid running out. */
  virtual void drop_file_handle() {}

  /* Name for logs and stats. */
  virtual string name() const = 0;

  /* Optional for tiled textures loaded externally. */
  virtual int get_tile_number() const;

  /* Free any memory used for loading metadata and pixels. */
  virtual void cleanup(){};

  /* Compare avoid loading the same image multiple times. */
  virtual bool equals(const ImageLoader &other) const = 0;
  static bool equals(const ImageLoader *a, const ImageLoader *b);

  virtual bool is_vdb_loader() const;

  /* Work around for no RTTI. */
};

/* Image Handle
 *
 * Access handle for image in the image manager. Multiple shader nodes may
 * share the same image, and this class handles reference counting for that. */
class ImageHandle {
 public:
  ImageHandle();
  ImageHandle(const ImageHandle &other);
  ImageHandle &operator=(const ImageHandle &other);
  ~ImageHandle();

  bool operator==(const ImageHandle &other) const;

  void clear();

  bool empty() const;
  int num_tiles() const;
  int num_svm_slots() const;

  ImageMetaData metadata();
  int svm_slot(const int slot_index = 0) const;
  vector<int4> get_svm_slots() const;

  device_image *vdb_image_memory() const;
  VDBImageLoader *vdb_loader() const;

  ImageManager *get_manager() const;

 protected:
  vector<size_t> slots;
  bool is_tiled = false;
  ImageManager *manager;

  friend class ImageManager;
};

/* Image Manager
 *
 * Handles loading and storage of all images in the scene. This includes 2D
 * texture images and 3D volume images. */
class ImageManager {
 public:
  explicit ImageManager(const DeviceInfo &info, const SceneParams &params);
  ~ImageManager();

  ImageHandle add_image(const string &filename, const ImageParams &params);
  ImageHandle add_image(const string &filename,
                        const ImageParams &params,
                        const array<int> &tiles);
  ImageHandle add_image(unique_ptr<ImageLoader> &&loader,
                        const ImageParams &params,
                        const bool builtin = true);
  ImageHandle add_image(vector<unique_ptr<ImageLoader>> &&loaders, const ImageParams &params);

  void device_update(Device *device, Scene *scene, Progress &progress);
  void device_free(Scene *scene);

  void device_load_builtin(Device *device, Scene *scene, Progress &progress);
  void device_free_builtin(Scene *scene);

  void device_load_slots(Device *device, Scene *scene, Progress &progress, const set<int> &slots);

  void device_update_requested(Device *device, Scene *scene);

  void set_osl_texture_system(void *texture_system);
  bool set_animation_frame_update(const int frame);

  void collect_statistics(RenderStats *stats);

  void tag_update();

  bool need_update() const;

  struct Image {
    ImageParams params;
    ImageMetaData metadata;
    unique_ptr<ImageLoader> loader;

    bool need_metadata;
    bool need_load;
    bool builtin;

    int users;
    thread_mutex mutex;

    // TODO: avoid storing these?
    uint texture_slot;
    uint tile_descriptor_offset;
    uint tile_descriptor_levels;
    uint tile_descriptor_num;
    device_image *vdb_memory;
  };

 private:
  bool need_update_;

  thread_mutex device_mutex;
  thread_mutex images_mutex;
  int animation_frame;

  vector<unique_ptr<Image>> images;
  ImageCache image_cache;
  void *osl_texture_system;

  bool use_texture_cache = true;
  bool auto_texture_cache = false;
  std::string texture_cache_path;

  size_t add_image_slot(unique_ptr<ImageLoader> &&loader,
                        const ImageParams &params,
                        const bool builtin);
  void add_image_user(const size_t slot);
  void remove_image_user(const size_t slot);
  Image *get_image_slot(const size_t slot);

  void load_image_metadata(Image *img);

  template<TypeDesc::BASETYPE FileFormat, typename StorageType>
  bool file_load_image(Device *device, Image *img, const int texture_limit);

  void device_load_image_tiled(Scene *scene, const size_t slot);
  void device_update_image_requested(Device *device, Scene *scene, Image *img);

  void device_load_image_full(Device *device, Scene *scene, const size_t slot);
  void device_load_image(Device *device, Scene *scene, const size_t slot, Progress &progress);
  void device_free_image(Scene *scene, const size_t slot);

  void device_resize_image_textures(Scene *scene);
  void device_copy_image_textures(Scene *scene);

  friend class ImageHandle;
};

CCL_NAMESPACE_END
