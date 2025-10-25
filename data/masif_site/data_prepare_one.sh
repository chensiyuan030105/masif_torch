#!/bin/bash
# Set the root directory of the MaSIF project manually
masif_root="/home/mhg/ForSiyuan/AlphaFold/masif_torch"  # Replace this with the actual path to your MaSIF project
masif_source=$masif_root/source/
masif_matlab=$masif_root/source/matlab_libs/
export PYTHONPATH=$PYTHONPATH:$masif_source
export masif_matlab

# Check if the PDB file already exists, if so, skip all steps and exit
if [ "$1" == "--file" ]
then
    echo "Running masif site on $2"
    PPI_PAIR_ID=$3
    PDB_ID=$(echo $PPI_PAIR_ID | cut -d"_" -f1)
    CHAIN1=$(echo $PPI_PAIR_ID | cut -d"_" -f2)
    CHAIN2=$(echo $PPI_PAIR_ID | cut -d"_" -f3)
    FILENAME=$2

    # Check if the PDB file already exists before copying and skip all if it does
    if [ -f "/home/mhg/ForSiyuan/AlphaFold/masif_torch/source/data_preparation/00-raw_pdbs/$PDB_ID.pdb" ]; then
        echo "[INFO] PDB file $PDB_ID.pdb already exists. Skipping all steps."
        exit 0
    else
        mkdir -p data_preparation/00-raw_pdbs/
        cp $FILENAME data_preparation/00-raw_pdbs/$PDB_ID.pdb
        echo "[INFO] PDB file copied: $PDB_ID.pdb"
    fi
else
    PPI_PAIR_ID=$1
    PDB_ID=$(echo $PPI_PAIR_ID | cut -d"_" -f1)
    CHAIN1=$(echo $PPI_PAIR_ID | cut -d"_" -f2)
    CHAIN2=$(echo $PPI_PAIR_ID | cut -d"_" -f3)

    # Check if the PDB file already exists and skip all if it does
    if [ -f "/home/mhg/ForSiyuan/AlphaFold/masif_torch/source/data_preparation/00-raw_pdbs/$PDB_ID.pdb" ]; then
        echo "[INFO] PDB file $PDB_ID.pdb already exists. Skipping all steps."
        exit 0
    else
        python -W ignore $masif_source/data_preparation/00-pdb_download.py $PPI_PAIR_ID
    fi
fi

# Process PDB chains
if [ -z $CHAIN2 ]
then
    echo "[INFO] CHAIN2 is empty. Processing only CHAIN1: $CHAIN1"
    echo "[INFO] Running 01-pdb_extract_and_triangulate.py for $PDB_ID_$CHAIN1"
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
else
    echo "[INFO] CHAIN2 is not empty. Processing CHAIN1: $CHAIN1 and CHAIN2: $CHAIN2"
    echo "[INFO] Running 01-pdb_extract_and_triangulate.py for $PDB_ID_$CHAIN1"
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN1
    echo "[INFO] Running 01-pdb_extract_and_triangulate.py for $PDB_ID_$CHAIN2"
    python -W ignore $masif_source/data_preparation/01-pdb_extract_and_triangulate.py $PDB_ID\_$CHAIN2
fi

echo "[INFO] Running 04-masif_precompute.py for masif_site with PPI_PAIR_ID: $PPI_PAIR_ID"
python $masif_source/data_preparation/04-masif_precompute.py masif_site $PPI_PAIR_ID
