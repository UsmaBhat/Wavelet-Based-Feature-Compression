##Dataset
- I have downloaded the Chaoyang dataset from here [HERE](https://bupt-ai-cz.github.io/HSA-NRL/).
-  Chaoyang dataset contains 1111 normal, 842 serrated, 1404 adenocarcinoma, 664 adenoma, and 705 normal, 321 serrated, 840 adenocarcinoma, 273 adenoma samples for training and testing, respectively. (Notes: "0" means normal, "1" means serrated, "2" means adenocarcinoma, and "3" means adenoma in our dataset files.)

## Citation
If you use this code for your research, please cite our paper ["Wavelet-Based Feature Compression for Improved Knowledge Distillation"].
```
@article{Usma,
  title={Wavelet-Based Feature Compression for Improved Knowledge Distillation},
  author={Niyaz, Usma and Sambyal, Abhishek Singh and Bathula, Deepti},
  journal={IEEE ISBI 2024}
}
```
## Overview of Proposed Framework
![model](https://github.com/UsmaBhat/Wavelet-Based-Feature-Compression-for-Improved-Knowledge-Distillation/assets/161701323/8b44e018-6ba2-4dc2-88af-9a44c18fe0c7)

## Comparison with SOTA
![vis3](https://github.com/UsmaBhat/Wavelet-Based-Feature-Compression-for-Improved-Knowledge-Distillation/assets/161701323/19d1a8b0-23f3-4561-90f2-0c99be3f22c8)


## Using instructions

- **Getting started:**

    Run `python main.py -a 0.1 -p 4 -e 200 -n 1 -r 1 -v V1 -w 1 -k 50`  train the model.
    1. alpha (-a): Weightage given to distillation in the loss function (e.g., -a 0.2).
    2. GPU (-p): GPU device selected for your program (e.g., -p 1 for GPU #1).
    3. epochs (-e): Number of epochs to run the program.
    4. #run (-n): Number of runs (e.g., -n 1 for the 1st run).
    5. run (-r): -r 1 for training, -r 0 for testing.
    6. version (-v): Choose between V1 (ResNet50-ResNet18) and V2 (ResNet50-MobileNetV2).
    7. run_wavelet (-w): Run the program with or without wavelet.
    8. keep (-keep): Percentage of coefficients to keep in wavelets.



## License

This project is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the code  given that you agree to our license terms below:

1. Make sure that you include a reference to our paper in any work that makes use of the code. For research papers, cite our preferred publications
2. That all rights not expressly granted to you are reserved by us.




## References
1. cite(https://github.com/bupt-ai-cz/HSA-NRL)



Usma Niyaz
- email: usma.20csz0015@iitrpr.ac.in


If you have any questions, please contact us directly.

