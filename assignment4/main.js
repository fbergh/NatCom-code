'use strict'
let sim;
let meter;


// Sets the parameters of the config object according to the on page inputs
function setParams() {
  config.simsettings.NRCELLS[0] = document.getElementById('numobst').value;
  config.simsettings.NRCELLS[1] = document.getElementById('numcells').value;
}

// Initialises the environment
function initialize() {
  killRunning();
  setParams();
  let custommethods = {
    initializeGrid : initializeGrid
  }
  sim = new CPM.Simulation(config, custommethods)
  meter = new FPSMeter({left:"auto", right:"5px"})
  step();
}

// Step function for the simulation
function step() {
  sim.step();
	meter.tick()
  requestAnimationFrame(step);
}

// Stops and removes a simulation if it exists
function killRunning() {
  if (sim !== undefined) {
    sim.toggleRunning();
    document.getElementsByTagName('canvas')[0].remove();
  }
}

function initializeGrid(){
  // add the GridManipulator if not already there and if you need it
  if (!this.helpClasses["gm"]){ this.addGridManipulator() }

  // Space obstacles evenly
  let nrObstacles = Number(this.conf.NRCELLS[0]);
  let pad = 20;
  let width = this.C.extents[0]; let height = this.C.extents[1];
  let xStep = (width -pad*2) / (Math.ceil(Math.sqrt(nrObstacles))-1);
  let yStep = (height-pad*2) / (Math.ceil(Math.sqrt(nrObstacles))-1);
  let seededObstacles = 0
  for (let y=pad; y<height; y+=yStep) {
    for (let x=pad; x<width; x+=xStep) {
      if (seededObstacles++ >= nrObstacles) 
        break;
      console.log(x);
      this.gm.seedCellAt(1, [Math.floor(x), Math.floor(y)]);
    }
  }

  // Randomly seed the moving cells
  for (let i=0; i<Number(this.conf.NRCELLS[1]); i++) {
    this.gm.seedCell(2)
  }
}
