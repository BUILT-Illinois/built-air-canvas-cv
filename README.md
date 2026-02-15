# B[U]ILT Air Canvas

## Getting Started
1. **Python3**: Download the latest version of python 3 at https://www.python.org/downloads/
2. Clone the repository on to your computer.
```
git clone https://github.com/BUILT-Illinois/built-air-canvas-cv.git
cd built-air-canvas-cv
```
3. Install dependencies from the requirements file.
```
pip3 install -r requirements.txt
```
Learn more about these libraries here:
- [Numpy](https://numpy.org)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV](https://opencv.org/releases/)

**Note:** On first run, the hand landmarker model file (`hand_landmarker.task`) will be automatically downloaded from Google's model repository.

## Our Tech Stack

 Our current tech stack includes the use of:

 1. Python
 2. Numpy
 3. MediaPipe
 4. OpenCV

## Best Practices

 To ensure our repository has less conflicts and to ensure everyone can develop fluidly we will implement best practices on branching, code reviews, descriptive naming/messages, modular reusable code, and commits.

 Please read up on these best practices as well, [What are Git version control best practices?](https://about.gitlab.com/topics/version-control/version-control-best-practices/)

### Branching

 Feature branching is a great way for teams to split up work and ensure there are reduced merge conflicts. This also ensures that the scope of pull requests are focused and specific. 

 A branch should be named with a proper name as well to signal to others what the purpose of the banch is, who is working on the branch, and where a feature will be located. 

 For example a branch for a rotating photo carousel feature on an about page by Steven can take on the form:  ***stevenuru/about/rotating-carousel***

### Commits

 Commits to a branch should be done granularly with frequency. Commiting large changes to a codebase can make it difficult for reviewers to gather a sense of what's going on and can make it more difficult to spot errors in one's code merge.

 Branch merges to main should also be reviewed by another committee member to reduce the risk of errors and to maintain a clean codebase.

### Writing Code

 #### Modularity
 
 Code should be written with reusability and modularity in mind. Functions should be used when possible to decrease repetititve code and makes it easier for reviewers to read when done right. Files should also be split up based on the functionality/purpose of the file. 

 #### Comments

 Comments should be used when the code itself cannot communicate to others what is happening. We should strive to use descriptive (although short) comments when defining new functions. This helps ramp up learning whenever someone new is introduced to the codebase
