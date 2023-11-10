
# An Evaluation Method for Temporal Spatial Sound Attributes

Official repository of the paper:

[Moiragias, G., & Mourjopoulos, J. (2023, May). An Evaluation Method for Temporal Spatial Sound Attributes. In Audio Engineering Society Convention 154. Audio Engineering Society.](http://www.aes.org/e-lib/browse.cfm?elib=22051)

The provided code implements the proposed methodology for assessing the agreement of a listening panel in temporal perceptual evaluation tasks.
A non-centered Principal Component Analysis is performed on the temporal ratings of all the assessors for each sound stimulus, and an agglomerative clustering algorithm is employed on the emerging eigenvectors to detect possible outliers, meaning temporal ratings that are not in agreement with those of the majority of the listening panel.

## Abstract
This work proposes a methodology for assessing temporal sound attributes, which can be applied to register time-varying perceptual attributes of audio signals reproduced via different spatial systems. The proposed methodology is inspired by state of the art temporal sensory evaluation of food, and such concepts are integrated into current spatial sound evaluation approaches. Here, as an example, we consider the dynamic evaluation of the perceptual attribute of envelopment, time intensity of envelopment (TIE), for music samples reproduced in three different spatial formats (mono, stereo and surround 5.0). The work describes a proposal for appropriate listening tests and data collection, along with a novel method for post-screening and extracting reliable temporal attributes from the listener assessments. A thorough description of each component of the listening test is given and a novel methodology is proposed for assessing the agreement of the listening panel and reducing the noise of the obtained data, based on non-centered Principal Component Analysis and hierarchical clustering algorithms.
## Requirements
You will need at least python 3.9. See requirements.txt for the required package versions.
## Usage
The module __panel_evaluation.py__ contains all the necessary functions and is accompanied with a thorough description of each function.
Function __agreement_evaluation__ is the main function, performing the non-centered PC decomposition of the temporal ratings and the outlier detection algorithm, and outputs the assessors that are in agreement and those who are not.
Function __remove_assessors__ removes the ratings of the inconsistent assessors from the csv that contains all the temporal ratings obtained during the listening test procedure.

The script __demo.py__ can be used for evaluating the efficacy of the proposed methodology on test data (__Demo_Temporal_Envelopment_Assessment.csv__) located in the folder __Temporal Envelopment Data__.
Running __demo.py__ will generate a new csv file on the same folder that contains only the ratings of the assessors in agreement, along with three figures (one for each of the three sound stimuli).
The figures depict the eigenvectors of the ratings of the assessors in the two-dimensional PC space and the respective temporal ratings. The ratings of the consistent assessors are noted in blue, while those of the inconsistent assessors in red. 

## Remarks
The proposed methodology was initially developed on temporal Envelopment ratings of music samples of duration of 15 seconds reproduced via loudspeakers in mono, stereo and multichannel (5.0) formats.
Adaptation of this methodology on other sound perceptual attributes or on audio signals of longer duration may be needed. 
