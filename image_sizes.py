import pyvips
import glob
from collections import defaultdict
from concurrent import futures as cf
from itertools import chain
from collections import Counter
import os
import statistics
import yaml
from yaml.representer import Representer
yaml.add_representer(defaultdict, Representer.represent_dict)

LAYER_STATS = defaultdict(dict)


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def fetch_sizes(tif_files,num_layers):
    layer_stats = defaultdict(list) # defaultdict uses more memory
    for _file in tif_files:
        for layer in range(num_layers):
            try:
                page = pyvips.Image.tiffload(_file, page=layer)
                width, height = page.width, page.height
                layer_stats[layer].append(max(width, height))
            except Exception as e:
                continue
    return layer_stats

def fetch_layers(tif_files):
    num_layers_list = []
    for _file in tif_files:
        image = pyvips.Image.tiffload(_file)
        num_layers_list.append(image.get_n_pages())
    return num_layers_list

def main():
    tif_files = glob.glob('/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source/images/*.tif')
    layer_stats = defaultdict(list)
    image = pyvips.Image.tiffload(tif_files[0])
    num_layers = image.get_n_pages()
    num_workers = 8
    worker_size = len(tif_files)//num_workers
    print("number of images:")
    print(len(tif_files))
    LAYER_STATS["n_images"] = len(tif_files)


    """Count distribution of number of layers per tif file"""
    with cf.ProcessPoolExecutor(max_workers = num_workers) as executor:
        futures_list = []
        for i in range(num_workers):
            start = i * worker_size
            stop = (i+1) * worker_size
            futures_list.append(executor.submit(fetch_layers,tif_files[start:stop]))
        cf.wait(futures_list)
        results = [future.result() for future in futures_list]

    merged_results = list(chain.from_iterable(results))
    print("number of layers in a tif file:")
    print(Counter(merged_results))

    for n_layer, _counts in dict(Counter(merged_results)).items():
        LAYER_STATS["images_n_layers_max"][n_layer] = _counts


    """Collect all the sizes that appear for a layer"""
    with cf.ProcessPoolExecutor(max_workers = num_workers) as executor:
        futures_list = []
        for i in range(num_workers):
            start = i * worker_size
            stop = (i+1) * worker_size
            futures_list.append(executor.submit(fetch_sizes,tif_files[start:stop],num_layers))

        cf.wait(futures_list)
        results = [future.result() for future in futures_list]

    layer_stats = {}
    for key in range(num_layers):
        layer_stats[key] = list(chain.from_iterable(d[key] for d in results))
    print("sizes per layer: ")

    """Summarize all the sizes that appear for a layer"""
    summary_stats = {}
    for layer, dimensions in layer_stats.items():
        summary_stats[layer] = {
            'min': min(dimensions),
            'max': max(dimensions),
            'mean': sum(dimensions)/len(dimensions),
            'median': statistics.median(dimensions),
        }
        LAYER_STATS["minimum_dimension_per_layer"][layer] = min(dimensions)
        LAYER_STATS["maximum_dimension_per_layer"][layer] = max(dimensions)
        LAYER_STATS["mean_dimension_per_layer"][layer] = sum(dimensions)/len(dimensions)
        LAYER_STATS["median_dimension_per_layer"][layer] = statistics.median(dimensions)

    print(summary_stats)
    # layer_stats = default_to_regular(LAYER_STATS)
    # print(layer_stats)


    with open('layer_descriptions.yml', 'w') as outfile:
        yaml.dump(LAYER_STATS, outfile, default_flow_style=False)

if __name__ == "__main__":
    main()

            # alternative
            # if layer not in layer_stats:
            #     layer_stats[layer] = []
            # layer_stats[layer].append((width, height))



# level = pyvips.Image.new_from_file("thing.tif", page=3)

# # Dictionary to store layer statistics
# layer_stats = {}

# # Iterate over each .tif file
# for file in tif_files:
#     # Load the .tif pyramid file
#     image = pyvips.Image.tiffload(file, n=-1)  # `n=-1` loads all layers/pages

#     # Get the number of layers/pages
#     num_layers = image.get("n-pages")
    
#     # Iterate through each layer/page
#     for layer in range(num_layers):
#         page = pyvips.Image.tiffload(file, page=layer)
#         width, height = page.width, page.height
        
#         # Record the dimensions
#         if layer not in layer_stats:
#             layer_stats[layer] = []
#         layer_stats[layer].append((width, height))

# # Calculate statistics
# summary_stats = {}
# for layer, dimensions in layer_stats.items():
#     widths = np.array([dim[0] for dim in dimensions])
#     heights = np.array([dim[1] for dim in dimensions])
    
#     summary_stats[layer] = {
#         'width': {
#             'min': widths.min(),
#             'max': widths.max(),
#             'mean': widths.mean(),
#             'median': np.median(widths),
#         },
#         'height': {
#             'min': heights.min(),
#             'max': heights.max(),
#             'mean': heights.mean(),
#             'median': np.median(heights),
#         }
#     }

# print(summary_stats)

