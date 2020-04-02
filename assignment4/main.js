'use strict'
let sim;
let meter;

function setParams() {
  let value = document.getElementById('numobst').value
  config.simsettings.NRCELLS[0] = value;
}

function initialize() {
  setParams();
  sim = new CPM.Simulation(config);
  meter = new FPSMeter({left:"auto", right:"5px"})
  step();
}

function step() {
  sim.step();
	meter.tick()
  requestAnimationFrame(step);
}
