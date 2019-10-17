#include "image2bayer.hpp"
#include "lodepng/lodepng.h"

#include <iostream>

using namespace std;

void decode(vector<uint8_t> &image, unsigned &width, unsigned &height,
            const char *filename) {
  // decode
  unsigned error = lodepng::decode(image, width, height, filename);

  // if there's an error, display it
  if (error)
    std::cout << "decoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;

  // the pixels are now in the vector "image", 4 bytes per pixel, ordered
  // RGBARGBA..., use it as texture, draw it, ...
}

void encode(const char *filename, std::vector<unsigned char> &image,
            unsigned width, unsigned height) {
  // Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);

  // if there's an error, display it
  if (error)
    std::cout << "encoder error " << error << ": " << lodepng_error_text(error)
              << std::endl;
}

#include "filters_definition.hpp"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cout << "Usage: image2bayer [filter type] [image.png] [output.png]" << endl;
    return -1;
  }

  int filter_type = atoi(argv[1]);
  char *im_file = argv[2];

  image im;
  decode(im.data, im.width, im.height, im_file);

  image raw;
  raw.width = im.width;
  raw.height = im.height;
  raw.data.resize(raw.width * raw.height * 4);

  image *filter = &filters[filter_type];

  for (uint32_t y = 0; y < im.height; y++) {
    for (uint32_t x = 0; x < im.width; x++) {
      color *c = im.getPixel(x, y);
      color *f = filter->getPixel(x % filter->width, y % filter->height);
      color filtered = *c * *f;
      raw.setPixel(x, y, &filtered);
    }
  }

  encode(argv[3], raw.data, raw.width, raw.height);

  char raw_filename[512];
  sprintf(raw_filename, "%s.bin", argv[2]);

  FILE* raw_file = fopen(raw_filename, "w");

  fwrite(&raw.width, sizeof(uint32_t), 1, raw_file);
  fwrite(&raw.height, sizeof(uint32_t), 1, raw_file);

  for (uint32_t y = 0; y < im.height; y++) {
    for (uint32_t x = 0; x < im.width; x++) {
      color *c = raw.getPixel(x, y);
      color *f = filter->getPixel(x % filter->width, y % filter->height);
      float luma = (float)(c->r + c->g + c->b) / (float)(f->r + f->g + f->b);
      fwrite(&luma, sizeof(float), 1, raw_file);
    }
  }

  fclose(raw_file);

  return 0;
}