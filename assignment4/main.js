'use strict'
let sim;

function setParams() {
  let value = document.getElementById('numobst').value
  config.simsettings.NRCELLS[0] = value;
}

function initialize() {
  setParams();
  sim = new CPM.Simulation(config);
  step();
}

function step() {
  sim.step();
  requestAnimationFrame(step);
}
