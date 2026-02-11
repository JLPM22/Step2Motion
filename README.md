
<div align="center">

# Step2Motion: Locomotion Reconstruction from Pressure Sensing Insoles <br> Eurographics 2026

[Jose Luis Ponton](https://joseluisponton.com/)<sup>1</sup>, [Eduardo Alvarado](https://edualvarado.com/)<sup>2</sup>, [Lin Geng Foo](https://lingeng.foo/)<sup>2</sup>, [Nuria Pelechano](https://www.cs.upc.edu/~npelechano)<sup>1</sup>, [Carlos Andujar](https://www.cs.upc.edu/~andujar)<sup>1</sup>, [Marc Habermann](https://people.mpi-inf.mpg.de/~mhaberma/)<sup>2</sup>

<sup>1</sup> [Universitat Politècnica de Catalunya (UPC)](https://www.upc.edu/en)  
<sup>2</sup> [Max Planck Institute for Informatics](https://www.mpi-inf.mpg.de/)

[**Project page**](https://vcai.mpi-inf.mpg.de/projects/Step2Motion/) | [**Paper (arXiv)**](https://arxiv.org/abs/2510.22712) | [**Video**](#) | [**Data**](#)

<img src="docs/teaser.jpg" alt="Step2Motion teaser" style="max-width: 100%;"/>

</div>

---

## Overview

Step2Motion is a system for reconstructing full-body locomotion from multi-modal insole sensors (pressure + IMU). It enables robust motion capture in unconstrained, real-world environments, without the limitations of traditional mocap suits or optical systems.

---

## Quick Start

> **Requirements:**  
> Python 3.9+  
> PyTorch (see [installation guide](https://pytorch.org/get-started/locally/))

1. Clone this repository.
2. Create and activate a virtual environment:
     ```bash
     python -m venv env
     .\env\Scripts\activate
     ```
3. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
4. Install PyTorch as per your system configuration.

---

## Example Usage

- **Process UnderPressure data:**
    ```bash
    python .\src\process_underpressure.py underpressure ..\data\UnderPressure\
    ```

- **Process MPI data (with optional Xsens retargeting):**
    ```bash
    python .\src\process_mpi.py mpi_dance ..\data\MPI\dancing\
    python .\src\process_mpi.py mpi_dance ..\data\MPI\dancing\ --xsens
    ```

- **Process "in-the-wild" data:**
    ```bash
    python .\src\process_insole_wild.py ..\evaluation\ling_geng_2024_12_11.txt .\skeletons\UPSkeleton_S1_AMASS.bvh
    ```

- **Train a model:**
    ```bash
    python .\src\train_no_prior.py --config .\configs\config_no_prior.json
    ```

- **Test a model:**
    ```bash
    python .\src\test.py .\models\no_prior_UP .\skeletons\UPSkeleton_S4_AMASS.bvh --dataset ..\data\UnderPressureTest\S4\s4_test.pt --clip 0
    ```

- **Visualize metrics:**
    ```bash
    python .\src\visualize_metrics.py
    ```

---

## Folder Structure

- `src/` — Main source code
- `configs/` — Configuration files and normalizers
- `models/` — Trained models
- `skeletons/` — Skeleton files (BVH)
- `Resources/` — Blender and other resources
- `Unity/` — Unity integration (if applicable)

---

## Citation

If you use this project, please cite:

```bibtex
@misc{ponton2025step2motion,
            title={Step2Motion: Locomotion Reconstruction from Pressure Sensing Insoles}, 
            author={Jose Luis Ponton and Eduardo Alvarado and Lin Geng Foo and Nuria Pelechano and Carlos Andujar and Marc Habermann},
            year={2025},
            eprint={2510.22712},
            archivePrefix={arXiv},
            primaryClass={cs.GR},
            url={https://arxiv.org/abs/2510.22712}, 
}
```

---

## License

Released under the MIT License. See `LICENSE` for details.

---