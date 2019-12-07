#include "libraw/libraw.h"
#include <fstream>
#include <iostream>

using namespace std;

void process_image(char *input_file, char *output_file) {
    LibRaw ip;
    ip.open_file(input_file);
    cout << ip.unpack() << endl;
    cout << ip.raw2image() << endl;

    libraw_data_t& idata = ip.imgdata;
    libraw_rawdata_t& raw = ip.imgdata.rawdata;

    auto rawimage = raw.raw_image;

    cout << idata.idata.make << " " << idata.idata.model << endl;
    cout << idata.sizes.raw_width << "x" << idata.sizes.raw_height << endl;

    cout << raw.raw_image << endl;
    cout << raw.color3_image << endl;
    cout << raw.color4_image << endl;
    cout << raw.float_image << endl;
    cout << raw.float3_image << endl;
    cout << raw.float4_image << endl;

    cout << rawimage[0] << endl;

    auto file = fstream(output_file, std::ios::out | std::ios::binary);
    file.write((char *)&idata.sizes.raw_width, sizeof(ushort));
    file.write((char *)&idata.sizes.raw_height, sizeof(ushort));
    file.write((char *)rawimage,
               sizeof(float) * idata.sizes.raw_width * idata.sizes.raw_height);
    file.close();
}

int main(int argc, char **argv) {
    if (argc != 3) return -1;

    process_image(argv[1], argv[2]);

    return 0;
}