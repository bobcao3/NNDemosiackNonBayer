#pragma once

#include "image2bayer.hpp"

std::vector<image> filters = {
    // Bayer
    {{/* R */ 255, 0, 0, 255, /* G */ 0, 255, 0, 255,
      /* G */ 0, 255, 0, 255, /* B */ 0, 0, 255, 255},
     /* width */ 2,
     /* height */ 2},
    // Quad Bayer
    {{/* R */ 255, 0,   0,   255, /* R */ 255, 0,   0,   255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* R */ 255, 0,   0,   255, /* R */ 255, 0,   0,   255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* B */ 0,   0,   255, 255, /* B */ 0,   0,   255, 255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* B */ 0,   0,   255, 255, /* B */ 0,   0,   255, 255},
     /* width */ 4,
     /* height */ 4},
    // X-Trans
    {{/* R */ 255, 0,   0,   255, /* B */ 0,   0,   255, 255,
      /* G */ 0,   255, 0,   255, /* B */ 0,   0,   255, 255,
      /* R */ 255, 0,   0,   255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* R */ 255, 0,   0,   255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* B */ 0,   0,   255, 255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* B */ 0,   0,   255, 255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* R */ 255, 0,   0,   255,
      /* B */ 0,   0,   255, 255, /* R */ 255, 0,   0,   255,
      /* G */ 0,   255, 0,   255, /* R */ 255, 0,   0,   255,
      /* B */ 0,   0,   255, 255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* B */ 0,   0,   255, 255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* R */ 255, 0,   0,   255,
      /* G */ 0,   255, 0,   255, /* G */ 0,   255, 0,   255,
      /* R */ 255, 0,   0,   255, /* G */ 0,   255, 0,   255,
      /* G */ 0,   255, 0,   255, /* B */ 0,   0,   255, 255},
     /* width */ 6,
     /* height */ 6}};