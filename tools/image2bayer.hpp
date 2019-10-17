#pragma once

#include <inttypes.h>
#include <vector>

using namespace std;

const float rdiv = 1.0 / 255.0;

struct color {
  uint8_t r, g, b, a;

  color operator*(color other) {
    return {(uint8_t)(other.r * r * rdiv), (uint8_t)(other.g * g * rdiv),
            (uint8_t)(other.b * b * rdiv), (uint8_t)(other.a * a * rdiv)};
  }
};

struct image {
  vector<uint8_t> data;
  uint32_t width, height;

  inline color *getPixel(int x, int y) {
    return (color *)&data[(y * width + x) * 4];
  }

  inline void setPixel(int x, int y, color *c) {
    *((color *)&data[(y * width + x) * 4]) = *c;
  }
};