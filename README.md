# DETECTIVE
Forecasting spatiotemporal social events has significant benefits for society to provide the proper amounts and types of resources to manage catastrophes and any accompanying societal risks. Nevertheless, forecasting event subtypes are far more complex than merely extending binary prediction to cover multiple subtypes because of spatial heterogeneity, experiencing a partial set of event subtypes, subtle discrepancy among different event subtypes, nature of the event subtype, spatial correlation of event subtypes. We present DeEp mulTi-task lEarning for Spatio-Temporal inCompleTe qualItative eVent forEcasting (DETECTIVE) framework to effectively forecast the subtypes of future events by addressing all these issues. This formulates spatial locations into tasks to handle spatial heterogeneity in event subtypes and learns a joint deep representation of subtypes across tasks. This has the adaptability to be used for different types of problem formulation required by the nature of the events. Furthermore, based on the ``first law of geography'', spatially-closed tasks share similar event subtypes or scale patterns so that adjacent tasks can share knowledge effectively. To optimize the non-convex and strongly coupled problem of the proposed model, we also propose algorithms based on the Alternating Direction Method of Multipliers (ADMM). Extensive experiments on real-world datasets demonstrate the model's usefulness and efficiency.

## Links
Paper: Chowdhury, Tanmoy, Yuyang Gao, and Liang Zhao. "Deep Multi-task Learning for Spatio-Temporal Incomplete Qualitative Event Forecasting." Submitted on IEEE Transactions on Knowledge and Data Engineering. At the second stage of revision.

## Instructions:
1. Main Model:
Use [run_model.py](/main/run_model.py) to run the model and [configure.py](/main/configure.py) to configure the model.

### Data: 
1. [Event Scale Forecasting](https://cs.emory.edu/~lzhao41/pages/dataset_pages/social_media_event_scale_forecasting_dataset.htm)
2. [Event Subtype Forecasting](https://cs.emory.edu/~lzhao41/pages/dataset_pages/social_media_event_subtype_forecasting_dataset.htm)
3. [civil_datasets](/data/civil_datasets): These datasets were obtained from eight different countries in Latin America, namely Argentina, Brazil, Chile, Colombia, Mexico, Paraguay, Uruguay, and Venezuela.
4. [flu_datasets](/data/flu_datasets):
The 2 datasets for influenza outbreaks in the U.S. use Twitter data as the data source. 

## Citation
If you use this work, please cite the following dissertation.

Chowdhury, Tanmoy. "Cross Domain Reasoning Based on Graph Deep Learning." PhD diss., George Mason University, 2023.
