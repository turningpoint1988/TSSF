# TSSF

**Hierarchical refinements of cis-regulatory inputs improve scalable gene expression prediction** <br/>
The flowchart of TSSF is displayed as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/TSSF/blob/main/figure/flowchart.jpg>
</p>

<h4 align="center"> 
Fig.1 Schematic overview of TSSF.
</h4>

## Prerequisites and Dependencies

- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [PyTorch 1.10](https://pytorch.org/)
- [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive)
- Python packages: biopython, scikit-learn, pyBigWig, h5py, scipy, pandas, matplotlib, seaborn

Install by runing:

```
pip install -r requirements.txt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
```

## Other Tools

- [MEME Suite](https://meme-suite.org/meme/doc/download.html): It assembles several methods used by this paper, including MEME-ChIP, TOMTOM and FIMO.
- [Bedtools](https://bedtools.readthedocs.io/en/latest/content/installation.html): It is a powerful toolset for genome arithmetic.
- [IGV(Integrative Genomics Viewer)](https://igv.org/): It is an interactive genome visualization tool.

## Competing Methods

- [Seq2Exp](https://github.com/divelab/AIRS/tree/main/OpenBio/Seq2Exp): A sequence-to-expression network designed to explicitly discover and extract regulatory elements that drive target gene expression, thereby enhancing the accuracy of the gene expression prediction.
- [CREaTor](https://github.com/DLS5-Omics/CREaTor): A deep neural network based on hierarchical attention that utilizes cCREs within open chromatin regions, combined with ChIP-seq data of TFs and histone modifications, to predict gene expression levels.
- [EPInformer](https://github.com/pinellolab/EPInformer): A Transformer-based framework that predicts gene expression by integrating promoter-enhancer interactions and their sequences, epigenomic signals, and chromatin contacts.
- [ScPGE](https://github.com/turningpoint1988/ScPGE): A scalable computational framework that predicts gene expression by assembling DNA sequences, TF binding scores, and epigenomic tracks of discrete cCREs into 3-dimensional tensors.

## Data Preparation

- Download [Human Genome hg38.fa](https://hgdownload.soe.ucsc.edu/downloads.html#human) and [mm10.fa](https://hgdownload.soe.ucsc.edu/downloads.html#mouse), and then put them into the `Genome` directory.
- Download [Human V29 annotation file](https://www.gencodegenes.org/human/release_29.html) and [Mouse M21 annotation file](https://www.gencodegenes.org/mouse/release_M21.html), and then put them into the `GENCODE` directory.
- Download all [Human cCREs (hg38) and Mouse cCREs (mm10)](https://screen.encodeproject.org/), and then put them into the `cCREs` directory. 
- Download [Experimental datasets](https://www.encodeproject.org) by using the following script:

```
cd /path/TSSF/code-data
bash download.sh
```

After finished, you can run the following script to prepare data:

```
cd /path/TSSF/code-data
bash data_pre.sh
```

## Data Construction 

 In our design concept, we extracted a fixed number of cCREs surrounding genes to predict gene expression levels. Since cCREs are discretely distributed on both sides of genes, we need to integrate DNA sequences and multi-type epigenomic signals on these cCREs. 

```
cd /path/TSSF
bash encode.sh
```

## Model Execution

We can run the two-stage frameworks from the scratch using the following script:

```
cd /path/TSSF
bash run.sh
```

The process consists of three stages, (1) ‘warm-up’: select the best-performing model to continue training; (2) 'training': train the selected model ; (3) 'testing':  test the model.


## Predictive Performance

We quantified the performance of all models across multiple cell types and tissues using Pearson correlation coefficients (Pearsonr) and mean absolute error (MAE) metrics.

<p align="center"> 
<img src=https://github.com/turningpoint1988/TSSF/blob/main/figure/performance.jpg>
</p>

<h4 align="center"> 
Fig.2 Overall performance of the two-stage frameworks.
</h4>

## cCRE Prioritization 

We can excute the task of active cCRE Prioritization using the following script:

```
cd /path/TSSF
python cCRE_priorization.py -r <> -t <>
```

| Arguments  | Description                                               |
| ---------- | -------------------------------------------------------   |
| -r         | The path of the project, e.g., ${HOME}/TSSF               |
| -t         | The name of a specific cell, e.g. K562                    |


The classification performance (PRAUC) of various methods across different distance groups in cCRE-gene interactions is as follows:

<p align="center"> 
<img src=https://github.com/turningpoint1988/TSSF/blob/main/figure/cCRE_priorization.jpg>
</p>

<h4 align="center"> 
Fig.3 The ability of the two-stage frameworks to prioritize cell-type-specific cCREs.
</h4>
