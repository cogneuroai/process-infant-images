# Meetings

## 01/17/2025

1. Frame count mismatch
2. Subtitle segments is not equal to frames annotated by the LLM
3. 3-5 mins of segments should be the ideal duration of an episode
4. Visual world is very stable (meaning things do not change of extended periods of time)
5. Frequency/duration of episode

## 01/01/2025

1. The annotation of all the frames is finished.
2. Results from all the databases is merged into a single master database, and it can be found in `/N/project/infant_image_statistics/annotation_results/master.db`

## 12/28/2024

1. Labelling process has been quite slow, with our current estimates it would take over 1000 hours to finish the annotation of the entire dataset (refer the [Video Frames](https://indiana-my.sharepoint.com/:x:/r/personal/demistry_iu_edu/Documents/Video%20frames.xlsx?d=w6415a16dccb944239204d0afc245d83b&csf=1&web=1&e=MSOy40) excel file).
2. The reason why this process could not be done in parallel is that, when multiple processes try to write in the database at the same time, it would lock the database and kill the annotation job.
3. Instead, to run things in parallel, every time we annotate a video, we are going to write results in a new database (with the same name as the video).
4. This will allow us to annotate multiple videos at the same time and finally once all the videos are annotated, we can merge the databases.
