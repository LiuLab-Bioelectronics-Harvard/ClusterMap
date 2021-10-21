"""
This file contains fuctions for reads assignment.
"""

# Load Packages
from .coding import *

import os
import tifffile
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.io import loadmat
from scipy.spatial import ConvexHull, cKDTree
from skimage.filters import laplace, gaussian
from scipy import ndimage as ndi
from skimage.morphology import binary_dilation, disk  # watershed
from skimage.segmentation import watershed
from skimage.transform import resize
from matplotlib.path import Path
from skimage.measure import regionprops
import matplotlib.patches as mpatches


# Get locations of nuclei from manual markers in DAPI channel
def parse_CellCounter(path):
    """
    Parse cell locations exported from Fiji CellCounter.
    Used to export manually selected cell locations from clicking on DAPI+ nuclei.
    """
    tree = ET.parse(path)
    root = tree.getroot()[1]
    vals = []
    for i, child in enumerate(root[1].findall("Marker")):
        x = int(child[0].text)
        y = int(child[1].text)
        vals.append([x,y])
    return np.array(vals)


# Preprocessing of Ilastik output
def load_ilastik_image(fpath):
    """
    Loads an Ilastik exported image, filters with gaussian, then thresholds.
    :param fpath: path of image
    :return: binary image where predicted > threshold
    """
    img = (tifffile.imread(fpath)-1.)*255.
    img = gaussian(img, sigma=2)
    return img > 250


# Load maximum projection of Nissl channel image
def load_nissl_image(dirname, fname="nissl_maxproj_resized.tif"):
    """
    Load Nissl data from directory containing nissl subdirectory.
    :param dirname: path/to/folder
    :param fname: image file name
    :return: numpy.ndarray
    """
    nissl = tifffile.imread(os.path.join(dirname, fname))
    return nissl


# Load label image
def load_label_image(dirname, fname="label_img.tif"):
    labels = tifffile.imread(os.path.join(dirname, fname))
    return labels


# Load segmentation image
def load_seg_image(dirname, fname="seg_img.tif"):
    seg = tifffile.imread(os.path.join(dirname, fname))
    seg = np.max(seg, axis=0)
    return seg


# Load cell locations from mat file
def load_cell_points(fpath):
    S = loadmat(os.path.join(fpath, "output", "cellLocs.mat"))
    return np.round(S["cellLocs"])


# Load reads and their positions from mat file
def load_read_position(fpath, reads_file):
    S = loadmat(os.path.join(fpath, reads_file))
    bases = [str(i[0][0]) for i in S["goodReads"]]
    points = S["goodPoints"][:, :2]
    temp = np.zeros(points.shape)
    temp[:, 0] = np.round(points[:, 1]-1)
    temp[:, 1] = np.round(points[:, 0]-1)
    return bases, temp


# Load gene table from genes.csv
def load_genes(fpath):
    genes2seq = {}
    seq2genes = {}
    with open(os.path.join(fpath, "genes.csv"), encoding='utf-8-sig') as f:
        for l in f:
            fields = l.rstrip().split(",")
            genes2seq[fields[0]] = "".join([str(s+1) for s in encode_SOLID(fields[1][::-1])])
            seq2genes[genes2seq[fields[0]]] = fields[0]
    return genes2seq, seq2genes


# Perform segmentation on nissl image
def segment_nissl_image(fpath, nissl, cell_locs, dilation=True):
    # apply gaussian filter & thresholding
    print("Gaussian & Thresholding")
    blurred_nissl_seg = gaussian(nissl.astype(np.float), 10) > 50
    if dilation:
        print("Dilating")
        blurred_nissl_seg = binary_dilation(blurred_nissl_seg, selem=disk(10))
    print("Assigning markers")
    markers = np.zeros(blurred_nissl_seg.shape, dtype=np.uint8)
    for i in range(cell_locs.shape[0]):
        y, x = cell_locs[i, :]
        if x < blurred_nissl_seg.shape[0] and y < blurred_nissl_seg.shape[1]:
            markers[x-1, y-1] = 1
    markers = ndi.label(markers)[0]
    print("Watershed")
    labels = watershed(blurred_nissl_seg, markers, mask=blurred_nissl_seg)
    labels_line = watershed(blurred_nissl_seg, markers, mask=blurred_nissl_seg, watershed_line=True)
    print(f"Labeled {len(np.unique(labels))} cells")

    out_path = os.path.join(fpath, "output")
    print(f"Saving files to {out_path}")
    tifffile.imsave(os.path.join(out_path, "labeled_cells_line.tif"), labels_line.astype(np.uint16))
    tifffile.imsave(os.path.join(out_path, "labeled_cells.tif"), labels.astype(np.uint16))
    return labels


# Assign reads to cells
def assign_reads_to_cells(labels, good_spots):
    Nlabels = labels.max()
    # make matrix of XY coordinates of label pixels and corresponding labels
    Npixels = len(np.where(labels > 0)[0])
    coords = []
    cell_ids = []
    print("Grabbing coordinates of cells xxx")
    num_cells = 0
    # for i in range(Nlabels):  # skip label 0 (background)
    #     curr_coords = np.argwhere(labels == i)
    #     if 100000 > curr_coords.shape[0] > 1000:
    #         coords.append(curr_coords)
    #         cell_ids.append(np.repeat(i, curr_coords.shape[0]))
    #         num_cells += 1
    #     else:
    #         coords.append(np.array([[], []]).T)

    for i, region in enumerate(regionprops(labels)):
        if 100000 > region.area > 1000:
            num_cells += 1
            coords.append(region.coords)
            cell_ids.append(np.repeat(i, region.coords.shape[0]))

    print("Using %d out of %d cells" % (num_cells, Nlabels))
    coords_list = coords
    coords = np.vstack(coords)
    cell_ids = np.concatenate(cell_ids)
    print("Building KD tree of cell coords")
    label_kd = cKDTree(coords)
    print("Assigning reads to cells")
    # print query_results[:10]
    cell_assignments = np.array([cell_ids[label_kd.query(p)[1]] for p in good_spots])
    return cell_assignments, coords_list


# Assign reads to the convex hull of each cell
def assign_reads_to_qhulls(labels, good_spots):
    # get convex hull of each cell
    num_cells = 0
    hulls = []
    coords = []

    print('Geting ConvexHull...')
    for i, region in enumerate(regionprops(labels)):
        if 100000 > region.area > 1000:
            num_cells += 1
            hulls.append(ConvexHull(region.coords))
            coords.append(region.coords)

    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1} cells")

    # reads assignment
    point_assignments = []
    print('Assigning reads to convex hull...')
    for i, h in enumerate(hulls):
        p = Path(h.points[h.vertices])
        point_assignments.append(np.argwhere(p.contains_points(good_spots)).flatten())
    return hulls, point_assignments, coords


def assign_reads_to_cells_3d(labels, good_spots):
    Nlabels = labels.max()
    print(f"Nlabels: {Nlabels}")

    # Npixels = len(np.where(labels > 0)[0])
    Nreads = good_spots.shape[0]
    good_spots = good_spots.astype(np.int)

    print("Getting labels of reads")
    # print(labels.shape)

    reads_label = []
    for i in range(Nreads):
        curr_loc = good_spots[i, :]
        # print(curr_loc)
        curr_label = labels[curr_loc[2], curr_loc[0], curr_loc[1]]  # ??
        reads_label.append(curr_label)

    return reads_label, Nlabels


# Reads quantification (convex hull)
def convert_reads_assignment_qhull(fpath, run_id, point_assignments, bases, seqs2genes):
    out_path = os.path.join(fpath, "output", run_id)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # construct expr matrix
    gene_seqs = seqs2genes.keys()
    Ncells = len(point_assignments)
    cell_by_barcode = np.zeros((Ncells, len(gene_seqs)))
    gene_seq_to_index = {}  # map from sequence to index into matrix

    for i, k in enumerate(gene_seqs):
        gene_seq_to_index[k] = i

    # print(gene_seq_to_index.keys())
    print("Counting reads...")
    total_read_count = 0
    for i in range(Ncells):
        if i % 50 == 0:
            print("Cell %d" % i)
        assigned_barcodes = point_assignments[i]  # which peaks are assigned to that cell
        for j in assigned_barcodes:  # which actual colorseq those correspond t
            b = bases[j]
            if b in gene_seq_to_index:
                cell_by_barcode[i, gene_seq_to_index[b]] += 1
                total_read_count += 1

    Ngood = float(len(bases))
    print("{:.2%} percent [{} out of {}] reads were assigned to cells".format(total_read_count/Ngood, total_read_count, Ngood))
    np.save(os.path.join(out_path, "cell_barcode_count.npy"), cell_by_barcode)
    np.savetxt(os.path.join(out_path, "cell_barcode_count.csv"), cell_by_barcode.astype(np.int), delimiter=',', fmt="%d")
    f = open(os.path.join(out_path, "cell_barcode_names.csv"), 'w')
    for i, k in enumerate(gene_seqs):
        f.write("%d,%s,%s\n" % (i, k, seqs2genes[k]))
    f.close()
    # return cell_by_barcode


# Reads qualification
def convert_reads_assignment(fpath, cell_assignments, bases, seqs2genes):
    outdir = os.path.join(fpath, "output", "singlecell")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    gene_seqs = seqs2genes.keys()
    # Nlabels = cell_assignments.flatten().max()
    good_cells = np.unique(cell_assignments)
    Ncells = good_cells.shape[0]
    cell_by_barcode = np.zeros((Ncells, len(gene_seqs)))
    gene_seq_to_index = {}

    for i, k in enumerate(gene_seqs):
        gene_seq_to_index[k] = i

    print(gene_seq_to_index.keys())
    print("Counting reads")
    total_read_count = 0
    for i in range(Ncells):
        if i % 50 == 0:
            print("Cell %d" % i)
        assigned_barcodes = np.where(cell_assignments == good_cells[i])[0]  # which peaks are assigned to that cell
        for j in assigned_barcodes:  # which actual colorseq those correspond t
            b = bases[j]
            cell_by_barcode[i, gene_seq_to_index[b]] += 1
            total_read_count += 1

    Ngood = float(len(bases))
    print("%f percent [%d out of %d] reads were assigned to cells" % (total_read_count/Ngood, total_read_count, Ngood))
    np.save(os.path.join(outdir, "cell_barcode_count.npy"), cell_by_barcode)
    np.savetxt(os.path.join(outdir, "cell_barcode_count.csv"), cell_by_barcode.astype(np.int), delimiter=',', fmt="%d")
    f = open(os.path.join(outdir, "cell_barcode_names.csv"), 'w')
    for i, k in enumerate(gene_seqs):
        f.write("%d,%s,%s\n" % (i, k, seqs2genes[k]))
    f.close()
    # return cell_by_barcode


# Convert reads assignment 3d
def convert_reads_assignment_3d(fpath, run_id, reads_label, Nlabels, bases, seqs2genes):
    out_path = os.path.join(fpath, run_id)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    reads_label = np.array(reads_label)

    gene_seqs = seqs2genes.keys()
    # print(seqs2genes)
    cell_by_barcode = np.zeros((Nlabels,len(gene_seqs)))
    # print(cell_by_barcode.shape)
    gene_seq_to_index = {}

    for i, k in enumerate(gene_seqs):
        gene_seq_to_index[k] = i

    # print(gene_seq_to_index.keys())
    print("Counting reads...")
    total_read_count = 0
    for i in range(1, Nlabels+1):
        # print(i)
        if i % 50 == 0:
            print("Cell %d" % i)
        assigned_barcodes = np.where(reads_label == i)[0]  # which peaks are assigned to that cell
        # print(np.where(reads_label==i))
        for j in assigned_barcodes:  # which actual colorseq those correspond t
            b = bases[j]
            cell_by_barcode[i-1, gene_seq_to_index[b]] += 1
            total_read_count += 1

    Ngood = float(len(bases))
    print("%f percent [%d out of %d] reads were assigned to cells" % (total_read_count/Ngood, total_read_count, Ngood))
    np.save(os.path.join(out_path, "cell_barcode_count.npy"), cell_by_barcode)
    np.savetxt(os.path.join(out_path, "cell_barcode_count.csv"), cell_by_barcode.astype(np.int), delimiter=',', fmt="%d")
    f = open(os.path.join(out_path, "cell_barcode_names.csv"), 'w')
    for i, k in enumerate(gene_seqs):
        f.write("%d,%s,%s\n" % (i, k, seqs2genes[k]))
    f.close()
    # return cell_by_barcode


# Make and save expression image
def save_expression_images(fpath, d, labels, hulls):
    outdir = os.path.join(fpath, "output", "singlecell")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    gene_names = d.columns
    for g in gene_names:
        img = make_expression_image(g, d, labels, hulls)
        ratio = 0.25
        row, col = img.shape
        plt.figure(figsize=(20, 20))
        plt.imshow(resize(img, (row * ratio, col * ratio)), cmap=plt.cm.get_cmap('jet'))
        plt.axis('off')
        plt.savefig(os.path.join(outdir, g+"_cells.png"))
        plt.close()


# Make expression image
def make_expression_image(gene_name, d, labels, hulls):
    Nlabels = len(hulls)
    expr = d[gene_name]
    expr_img = np.zeros_like(labels)
    for i in range(Nlabels):
        p = hulls[i].points.astype(np.int)
        expr_img[p[:, 0], p[:, 1]] = expr[i]
    return expr_img


# Plot a cell segmentation image with cell numbers
def plot_cell_numbers(fpath, labels):
    out_path = os.path.join(fpath, "output")
    plt.figure(figsize=(20, 10))
    plt.imshow(labels, cmap=plt.cm.get_cmap('jet'))
    for i, region in enumerate(regionprops(labels)):
        plt.text(region.centroid[1], region.centroid[0], str(i), fontsize=7, color='w')
    plt.savefig(os.path.join(out_path, "cell_nums.png"))


# Plot a cell segmentation image with cell numbers
def plot_cell_numbers_3d(fpath, labels):
    out_path = os.path.join(fpath, "output")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')

    for i, region in enumerate(regionprops(labels)):
        ax.scatter(region.centroid[1], region.centroid[0], region.centroid[2], marker='o', color='b')
        ax.text(region.centroid[1], region.centroid[0], region.centroid[2], str(i), fontsize=7, color='r')
    plt.savefig(os.path.join(out_path, "cell_nums.png"))


# Plot cluster
def plot_clusters(fpath, labels, hulls, ident_name, outname, cmap=None):
    outpath = os.path.join(fpath, "output")
    # load cluster labels
    num2ident = {}
    max_ident = 0
    with open(os.path.join(outpath, ident_name)) as f:
        for i, l in enumerate(f):
            if i > 0:
                name, ident = l.rstrip().split(",")
                cell_num = int(name.split("_")[1])+1
                num2ident[cell_num] = int(ident)
                if int(ident) > max_ident:
                    max_ident = int(ident)
    cluster_img = np.zeros_like(labels)
    for k, v in num2ident.items():
        p = hulls[k-1].points.astype(np.int)
        cluster_img[p[:, 0], p[:, 1]] = v+1
    plt.figure(figsize=(20,10))
    if cmap is None:
        cmap = plt.cm.get_cmap('OrRd')
    values = range(cluster_img.max() + 1)  # [0,1,2,3,4,5]
    im = plt.imshow(cluster_img, cmap=cmap, vmin=0, vmax=max_ident+1)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=values[i]-1)) for i in range(1, len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis('off')
    plt.savefig(os.path.join(outpath, outname), transparent=True)
