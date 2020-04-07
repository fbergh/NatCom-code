let volume_normal_cell = 200;
let config = {

  // Grid settings
  ndim : 2,
  field_size : [200,200],

  // CPM parameters and configuration
  conf : {
    torus : [true,true], // Should the grid have linked borders?
    // seed : 1337,  // Seed for random number generation.
    T : 20, // CPM temperature
    framerate: 10,

    // Adhesion parameters:
    J: [[0,20,0],
			[20,20,20],
			[20,20,20]],

    // VolumeConstraint parameters
    LAMBDA_V : [0,100,50], // VolumeConstraint importance per cellkind
    V : [0,volume_normal_cell/2,volume_normal_cell], // Target volume of each cellkind

    // Scale the parameters with the cell size such that behaviour is consistent across all cell sizes
    LAMBDA_P : [0,16*(volume_normal_cell/100),1*(volume_normal_cell/100)],
    P : [0,32*(volume_normal_cell/100),90*(volume_normal_cell/100)],

    LAMBDA_ACT : [0,0,200],  // ActivityConstraint importance per cellkind
    MAX_ACT : [0,0,80],  // Activity memory duration per cellkind
    ACT_MEAN : "geometric",
  },

  // Simulation setup and configuration
  simsettings : {
    // Cells on the grid
    NRCELLS : [2,2], // Number of cells to seed for all
    // non-background cellkinds.

    IMGFRAMERATE : 20,
    RUNTIME : 500, // Only used in node

    CANVASCOLOR : "eaecef",
    CELLCOLOR: ['000000','6305fb'],
    SHOWBORDERS : [false, true],
    ACTCOLOR : [false,false],
    zoom : 3 // zoom in on canvas with this factor.
  }
}
