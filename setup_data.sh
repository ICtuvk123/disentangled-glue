#!/bin/bash
# One-shot data + genome + tool setup for Muto-2021 and Yao-2021
set -e

REPO=/data1/users/zhutianci/proj/disentangled-glue
GENOME=$REPO/data/genome
MUTO=$REPO/data/download/Muto-2021
YAO=$REPO/data/download/Yao-2021

cd $REPO

# ─────────────────────────────────────────────────────────────────────────────
# 1. Genome annotations (GENCODE)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Downloading GENCODE GTFs ==="
cd $GENOME

[ -f gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz ] || \
    wget -c "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_35/gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz"

[ -f gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz ] || \
    wget -c "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Blacklist regions
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Cloning Blacklist ==="
cd $GENOME
if [ ! -d Blacklist ]; then
    git clone https://github.com/Boyle-Lab/Blacklist.git
    cd Blacklist
    git checkout c4b5e42b0c4d77ed4f1d1acd6bccd1297f163069
    cd ..
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Gene promoter BED files (needed by Muto preprocess.sh)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Building genes_with_promoters BED files ==="
cd $GENOME
[ -f gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed ] || {
    python extract_genes_promoters.py \
        --input-gtf gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz \
        --output-bed gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.bed
    LC_ALL=C sort -k1,1 -k2,2n -k3,3n \
        gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.bed \
        > gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. AMULET
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Installing AMULET ==="
if [ -z "$AMULET_HOME" ] || [ ! -f "$AMULET_HOME/AMULET.sh" ]; then
    cd $REPO
    if [ ! -d AMULET ]; then
        wget -c "https://github.com/UcarLab/AMULET/releases/download/v1.1/AMULET-v1.1.zip" -O AMULET.zip
        unzip -o AMULET.zip -d AMULET_tmp
        mv AMULET_tmp AMULET
        rm -rf AMULET_tmp AMULET.zip
        chmod +x AMULET/AMULET.sh
    fi
    export AMULET_HOME=$REPO/AMULET
    echo "export AMULET_HOME=$REPO/AMULET" >> ~/.bashrc
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Muto-2021: metadata h5ads from CellxGene
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Downloading Muto-2021 metadata h5ads ==="
cd $MUTO
[ -f rna.h5ad ] || \
    wget -c "https://datasets.cellxgene.cziscience.com/ff2e21de-0848-4346-8f8b-4e1741ec4b39.h5ad" \
         -O rna.h5ad
[ -f atac.h5ad ] || \
    wget -c "https://datasets.cellxgene.cziscience.com/43513175-baf7-4881-9564-c4daa2416026.h5ad" \
         -O atac.h5ad

# ─────────────────────────────────────────────────────────────────────────────
# 6. Muto-2021: GEO raw data
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Downloading Muto-2021 GEO raw data ==="
cd $MUTO
[ -f GSE151302_RAW.tar ] || \
    wget -c "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE151nnn/GSE151302/suppl/GSE151302_RAW.tar"

echo "=== Extracting and preprocessing Muto-2021 ==="
[ -f GSM4572192_Control1_filtered_feature_bc_matrix.h5 ] || tar xf GSE151302_RAW.tar
bash preprocess.sh

# ─────────────────────────────────────────────────────────────────────────────
# 7. Yao-2021: NeMO archive
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Downloading Yao-2021 from NeMO Archive ==="
cd $YAO
[ -f MOp_MiniAtlas_2020_bdbag_2021_04_28.tgz ] || \
    wget -c "https://data.nemoarchive.org/publication_release/MOp_MiniAtlas_2020_bdbag_2021_04_28.tgz"

[ -d MOp_MiniAtlas_2020_bdbag_2021_04_28 ] || \
    tar xzf MOp_MiniAtlas_2020_bdbag_2021_04_28.tgz

# ─────────────────────────────────────────────────────────────────────────────
# 8. Run collect scripts
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Running Muto-2021 collect ==="
cd $REPO/data/collect
python Muto-2021.py

echo "=== Running Yao-2021 collect ==="
python Yao-2021.py

echo "=== All done ==="
