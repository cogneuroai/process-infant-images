# Meetings

## 12/28/2024

1. Labelling process has been quite slow, with our current estimates it would take over 1000 hours to finish the annotation of the entire dataset (refer the [Video Frames](https://indiana-my.sharepoint.com/:x:/r/personal/demistry_iu_edu/Documents/Video%20frames.xlsx?d=w6415a16dccb944239204d0afc245d83b&csf=1&web=1&e=MSOy40) excel file).
2. The reason why this process could not be done in parallel is that, when multiple processes try to write in the database at the same time, it would lock the database and kill the annotation job.
3. Instead, to run things in parallel, every time we annotate a video, we are going to write results in a new database (with the same name as the video).
4. This will allow us to annotate multiple videos at the same time and finally once all the videos are annotated, we can merge the databases.