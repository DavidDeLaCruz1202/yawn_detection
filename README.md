# yawn_detection
A project based on a funny idea a professor had, and that I decided to implement in Python.

## Dataset Note
I'm not including the yawning dataset that I'm using, but it's easily found on Kaggle.

## Updates
1) 5/5/2025
- I finished making the classification model to detect when there isn't and is a yawn on the screen. I ran it twice, and the first final accuracy score was ~80% and the second was ~90%. I don't want to overtrain it so I'll play around with the layers and epochs and such to see if I can get it to maybe around 80% - 85% consistently to be more usable. After this, I need to figure out how to save this model, use a library to have the script use my camera to take screenshots and run this through the saved model, and then to output whether or not a yawn is being detected.