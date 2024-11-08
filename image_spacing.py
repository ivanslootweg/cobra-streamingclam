import pyvips
import glob
import os


def main():
    tif_files = glob.glob(r'/home/ivanslootweg/data/SBCC/images/*.tif')
    print(len(tif_files)) 
    _keys = ["width",  "height", "xoffset", "yoffset", "xres", "yres","resolution-unit"]
    for _file in tif_files:
        for layer in range(8):
            try:
                page = pyvips.Image.tiffload(_file, page=layer)
                metadata = page.get_fields()
                for _key in _keys:
                    print(_key, page.get(_key))
            except Exception as e:
                print(e)
                continue
        break


if __name__ == "__main__":
     main()
