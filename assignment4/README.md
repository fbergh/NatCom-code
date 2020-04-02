# Assignment 4

This is the code for assignment 4 regarding CPM. To run the simulation open `index.html` in Firefox/Chrome and click *Start*.

Explanation for each file:
- `index.html`: Main page. Taken from the boilerplate code and it includes the main structure of the page.
- `config.js`: The default config for this kind of experiments. This config is also modified using the controls on the page
- `main.js`: All the main code used for this assignment. It mainly calls the CPM library
- `style.css`: Some rules to make the page look kinda pretty.


## Questions
Some draft answers for the three questions of the description
### Question 1
Since everything in CPM is a cell, each obstacle should be a cell. All obstacles can be of the same "type".  Making a kind of round obstacle cell/can be achieved with these parameters:
```json
conf : {
  T : 20, // CPM temperature
  J: [[0,0], [0,20]] ,
  LAMBDA_V : [0,100],
  V : [0,153],
  P : [0,43],
  LAMBDA_P : [0,100]
},
```
In other words we set volume to 153 and perimeter to 43 and give them an equal lambda weight. The two numbers were found using the formula for the area and circumference of a circle with radius 7. As you can see the obstacle is not always round.  
