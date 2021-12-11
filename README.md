# Beta-Mapping
## Introduction

This repository contains code to run *Beta-Mapping*, an experimental method for constrained reinforcement learning.

Note that this project is still in the experimental stage and the code is not guaranteed to work.

We define a reinforcement learning policy over 
a known safe region <img src="https://render.githubusercontent.com/render/math?math={C}(s)">. 
Note that it is necessary that the region 
<img src="https://render.githubusercontent.com/render/math?math={C}(s)">
 is convex; in this work, we also assume that 
 <img src="https://render.githubusercontent.com/render/math?math={C}(s)">
  is defined by a set of inequalities, linear in the action 
  <img src="https://render.githubusercontent.com/render/math?math=a">, and is bounded. 
  In environments with bounded action spaces, augmenting the set of inequalities with the action space bounds easily satisfies this latter requirement. 

The basic idea to define a policy over a safe region is to select actions from the beta distribution, defined over 
<img src="https://render.githubusercontent.com/render/math?math=[0,1]^n">. 
Actions are then mapped to an action
<img src="https://render.githubusercontent.com/render/math?math=a^{'}">
in the cube 
<img src="https://render.githubusercontent.com/render/math?math=D\equiv p + \alpha[-1,1]^n">
that is a subset of the safe region 
<img src="https://render.githubusercontent.com/render/math?math={C}(s)">. Finally, actions are scaled from 
<img src="https://render.githubusercontent.com/render/math?math=D">
 to 
 <img src="https://render.githubusercontent.com/render/math?math={C}(s)">
along the line 
<img src="https://render.githubusercontent.com/render/math?math=\xi(t)= p+t (a'-p)">
 interpolating
 <img src="https://render.githubusercontent.com/render/math?math=a^{'}"> 
 and 
 <img src="https://render.githubusercontent.com/render/math?math=p"> 
 with the safe region action 
 <img src="https://render.githubusercontent.com/render/math?math=a^{''}=\xi(t^*)">
  depending on the factor 
  <img src="https://render.githubusercontent.com/render/math?math=t^*">
   that ensures that the distance between 
   <img src="https://render.githubusercontent.com/render/math?math=a^{''}">
    and the intersection of 
    <img src="https://render.githubusercontent.com/render/math?math=\xi(t)">
     with the boundary of 
     <img src="https://render.githubusercontent.com/render/math?math={C}(s)">
      is proportional to the distance between 
      <img src="https://render.githubusercontent.com/render/math?math=a^{'}">
       and the intersection of 
       <img src="https://render.githubusercontent.com/render/math?math=\xi(t)">
        with the boundary of 
        <img src="https://render.githubusercontent.com/render/math?math=D">
        . An example of this mapping is shown in the figure below.

Note that, crucially, the mapping 
<img src="https://render.githubusercontent.com/render/math?math=\xi(t^*)">
 is bijective -- each action 
 <img src="https://render.githubusercontent.com/render/math?math=a"> 
 selected by the agent from the beta-parameterized policy is mapped to a unique action 
 <img src="https://render.githubusercontent.com/render/math?math=a^{''}">
  in the constrained region. Because of this bijection, there is no bias in projection: 
  <img src="https://render.githubusercontent.com/render/math?math=Q_\pi(s,a^{''})-Q_\pi(s,a)=0">.

This repository contains code from the [Safe Explorer project](https://github.com/AgrawalAmey/safe-explorer),
in the safe_explorer folder.

![figure](beta_constraints/images/beta_mapping_example.png)