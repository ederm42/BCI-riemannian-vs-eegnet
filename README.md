# Benchmarking Brain-Computer Interface Algorithms using MOABB: Riemannian Methods vs. Convolutional Neural Networks 
This work uses an adapted fork of MOABB v0.4.6 and makes use of the EEGNet implementation from https://github.com/vlawhern/arl-eegmodels

## Abstract
*Objective.* To date, a comprehensive comparison of Riemannian decoding methods with deep convolutional neural networks for EEG-based Brain-Computer Interfaces remains absent from published work. We address this research gap by using MOABB, The Mother Of All BCI Benchmarks, to compare novel convolutional neural networks to state-of-the-art Riemannian approaches across a broad range of EEG datasets, including motor imagery, P300, and steady-state visual evoked potentials paradigms. 

*Approach.* We systematically evaluated the performance of convolutional neural networks, specifically EEGNet, shallow ConvNet, and deep ConvNet, against well-established Riemannian decoding methods using MOABB processing pipelines. This evaluation included within-session, cross-session, and cross-subject methods, to provide a practical analysis of model effectiveness and to find an overall solution that performs well across different experimental settings.

*Main results.* We find no significant differences in decoding performance between convolutional neural networks and Riemannian methods for within-session, cross-session, and cross-subject analyses.  

*Significance.* The results show that, when using traditional Brain-Computer Interface paradigms, the choice between CNNs and Riemannian methods may not heavily impact decoding performances in many experimental settings. These findings provide researchers with flexibility in choosing decoding approaches based on factors such as ease of implementation, computational efficiency or individual preferences. 
