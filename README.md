# Overview of Spiking Neural Networks frameworks used for object detection

#### Authors
- Maria Krzyżowska
- Julia Wiktorowicz
- Patrycja Wybranowska

#### Report available [here](https://www.overleaf.com/read/jndkpbfbqgbs#136ad1)


## Project overview
The aim of the project is to compare three PyTorch-based frameworks for designing, trainig and simulating Spiking Neural Networks (SNNs): snnTorch, Norse and SpikingJelly. The developed networks will be evaluated for object detection conducted on an automotive event-based dataset, with performance benchmarks conducted via NeuroBench framework. The most effective architecture will be optimized for deployment on real-time data stream directly from an event camera.  

## 6.05.2025
- [Literature review](https://github.com/pwybranowska/DVS-Overview-of-SNN-frameworks/blob/main/SNN_overwiew.pdf)
- Datasets overview
  - [Prophesee's GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)
- Selection of methods for further work
- Plan for the next milestone:
  - Choose frameworks for comparison (everyone gets one to implement)
  - Theoretical overview/description of each framework
  - criteria for comparison


## 13.05.2025
- Frameworks for comparison:
  - snnTorch (PW)
  - Norse (JW)
  - SpikingJelly (MK)
- Theoretical overview/description of each framework (report)
- Comparison method: [NeuroBench](https://neurobench.ai/)
- Plan for the next milestone:
  - Implementation of SNNs
  

## 20.05.2025
- Implementation of SNNs and training
- Comparing the differences in implementation
- Plan for the next milestone:
  - Comparing networks using NeuroBench (PW)
  - Performing evaluation (JW)
  - Implementing predictions comparisson (MK)

## 27.05.2025
- Performing evaluation
    * cls_loss (Classification Loss) - measures how accurately the model identifies what type of object is inside a bounding box (e.g., distinguishing a pedestrian from a car). A lower score means the model rarely misclassifies objects or confuses them with the background.
    * box_loss (Bounding Box Loss) - measures how precisely the model predicts where an object is by comparing the coordinates of its predicted boxes against the true boundaries. A lower score means the drawn boxes highly overlap and tightly fit the actual shapes of the objects.
    * loss (Total Loss) - the overall error metric calculated by summing together the classification and bounding box losses. It serves as the single main score the system tries to minimize during training to improve overall detection performance.
    * results saved in models-evaluation-results.txt
 
  
![czikapu](https://i.imgflip.com/4367fq.gif)

