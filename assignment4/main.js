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

  // Space obstacles kinda evenly
  let nrObstacles = Number(this.conf.NRCELLS[0]);
  let xStep = (this.C.extents[0]-20) / Math.sqrt(nrObstacles);
  let yStep = (this.C.extents[1]-20) / Math.sqrt(nrObstacles);
  let seededObstacles = 0
  for (let i=10; i<this.C.extents[0]-10 && seededObstacles < nrObstacles; i+=xStep) {
    for (let j=10; j<this.C.extents[1]-10 && seededObstacles < nrObstacles; j+=yStep) {
      this.gm.seedCellAt(1, [Math.floor(j), Math.floor(i)]);
      seededObstacles++;
    }
  }

  // Randomly seed the moving cells
  for (let i=10; i<Number(this.conf.NRCELLS[1]); i++) {
    this.gm.seedCell(2)
  }
}
