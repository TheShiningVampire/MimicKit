# Narration script with scene start times

## Title  (00:00)

In the last video, a simulated body learned to copy a human motion clip.
A back flip, a cartwheel, a spin kick.
This video asks a harder question.
What changes when the body must pick up a cup?

## Recap  (00:13)

Here is the recipe that MimicKit uses for whole body motion.
A recording of a person gives a reference pose at each instant of time.
A physics simulator holds the body, and a neural network policy drives the joints.
At each step the reward measures how close the simulated pose is to the reference pose.
Reinforcement learning then pushes the policy toward larger rewards.
The body copies the clip, and physics makes the copy real.

## WhatChanges  (00:39)

Now put a cup on the table.
The pose of the body is no longer the goal.
The goal is the cup.
If the hand is a bit off but the cup gets to the shelf, the task is done.
If the hand matches the recording perfectly but the cup falls, the task has failed.
So the reward must move from the body to the object.

## Contacts  (00:59)

The second difference is contact.
When a body walks, it touches the world at one or two flat feet, in a steady rhythm.
When a hand grasps, five finger tips touch a curved surface, all at slightly different times.
A one millimetre error in a step is nothing.
A one millimetre error in a pinch is the difference between a grasp and a drop.
So the simulation, the tracking, and the rewards must all become much finer.

## Observation  (01:24)

The third difference is what the policy must see.
A walking policy can get by with its own joint angles and a direction.
A grasping policy needs the object: where it is, how it is turned, and how it moves, relative to the hand.
It also helps to show it a short piece of the future reference, so that it can prepare the fingers before contact.

## Data  (01:45)

The fourth difference is the data.
Whole body motion capture has been standard for decades, and MimicKit ships with hundreds of clean clips.
Hand and object capture is harder: fingers hide each other, and the object hides the fingers.
The best data sets use many cameras, gloves, or marker suits, and record the object pose too.
GRAB, ARCTIC, OakInk2, HOI4D and EgoDex are the ones to start with.
Each gives, per frame, a body or hand pose, and a six D object pose.

## Retarget  (02:16)

A human hand and a robot hand are not the same shape.
So the recorded hand is retargeted.
An optimizer moves the robot joints until its finger tips sit where the human finger tips sat, relative to the object.
Whole body retargeting works the same way, but the object anchors everything.
Keep the object fixed, and move the body to fit it.

## Pipeline  (02:37)

Put it together and the recipe looks like this.
One, a human data set with object poses.
Two, retarget to the robot and write a MimicKit motion file that also carries the object trajectory.
Three, a new environment that spawns the object, sees it, and rewards tracking it.
Four, train with the same reinforcement learning loop as before.
Five, distill many clips into one policy, and only then think about the real robot.

## Plan  (03:05)

The plan has four phases.
Phase zero reproduces an existing method on its own data.
Phase one adds objects to MimicKit and tracks a single clip.
Phase two scales to a data set and a robot hand.
Phase three goes to the real robot.
Each phase has a clear test, so that a failure shows early.

## Close  (03:25)

Whole body mimicry taught the body to move like us.
Manipulation mimicry must teach it to change the world like us.
Same loop, new reward, much finer contact.
The full report, the plan, and the data samples are in the repository.

