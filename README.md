# Developing a KPI to the Slip-to-Slip Connection Time in a Drilling Operation
## Summary 

I delivered here a complete data-driven solution to monitor crew performance in drilling operations from an Oil & Gas company. The performance is measured according to the time taken to perform the slip-to-slip connection.

After a careful exploratory data analysis, I selected relevant measurements that come from a variety of sensors to create a machine learning model that is able to reliably categorize whether the slips are holding the upper part of a drill pipe. Since sensors give the timestamp for each measurement, after predicting the slips state, we can calculate the time it took for the crew to complete the process of connecting pipes.

This solution provides an automated way to monitor the crew performance saving the company time, and money.

## Introduction

Measurement of performance is vital to business success and most performance studies are
related to the so-called key performance indicators (KPI). A KPI measures how well the
organization is performing on operational, tactical or strategic activities. This is critical for the
current and future success of the organization.

In the Oil and Gas industry, a KPI can be used to monitor and measure operational quality and
crew performance. In a Drilling operation, the process where a hole is bored using a drill bit to
create a well for oil and natural gas production, an example of KPI is the Slip to Slip connection
time. This KPI can offer significant improvements in identifying sources of Non-Productive time.

The slip is a device used to grip and hold the upper part of a drill pipe to the drill floor, which is
the area where the pipe begins its trip into the earth. The Slips are used when making a
connection: the pipes are joined in order to advance further into the hole. Therefore, each pipe
is picked up by a hook, temporarily gripped by the slips and then joined to another pipe. After
the joint, the slips are removed and the entire pipe is carefully lowered into the hole, resuming
the drilling. A skilled rig crew can physically accomplish all of those steps in a minute or two.

In this challenge, my goal is to perform an Exploratory Data Analysis to extract useful
information for the development of the above-mentioned KPI. Further, I need to create an ML model to monitor the performance of the rig crew in a drilling operation. 

## Concluding Remarks

After performing an exploratory data analysis, the statistical metrics showed us which features are the most important considering the goal of this project. Subsequently, my findings were confirmed by the relevance analysis performed by machine learning; the features TOR, DEPT, and RPM can be removed to simplify the final model.

Besides that, the correlation analysis told us that most features are extremely uncorrelated, which gave me an idea of which algorithms I should select to start the modelling. The finds suggested that tree-based models would be a good fit for the task, which was confirmed by the classification reports.

In summary, the supervised learning algorithm LGBMClassifier outperformed all the other tested algorithms including the unsupervised-learning one K-Means Clustering. From the evaluation metrics, we can safely say that, considering the given data, the model can accurately capture the operational state of the slips with 99% precision. The estimated average connection time deviates only 10% from the connection time calculated with the labelled data.

Finally, the developed solution seems to be reliable and can be used for automated monitoring of performance on drilling operations for any O&G industry.

### Possible Next Steps
As a possible next step, if I had more time, I'd put this model in production on a cloud environment so that it can receive requests from other applications to stream predictions and deliver the wanted KPI. 

Additionally, I'd like to evaluate how this model would perform on previously unseen data.
