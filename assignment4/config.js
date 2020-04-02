let config = {

	// Grid settings
	ndim : 2,
	field_size : [200,200],

	// CPM parameters and configuration
	conf : {
		torus : [true,true], // Should the grid have linked borders?
		seed : 1337,  // Seed for random number generation.
		T : 20, // CPM temperature

		// Adhesion parameters:
		J: [[0,0], [0,20]] ,

		// VolumeConstraint parameters
		LAMBDA_V : [0,100], // VolumeConstraint importance per cellkind
		V : [0,153], // Target volume of each cellkind
  	P : [0,43],
  	LAMBDA_P : [0,100]
	},

	// Simulation setup and configuration
	simsettings : {
		// Cells on the grid
		NRCELLS : [300], // Number of cells to seed for all
		// non-background cellkinds.

		RUNTIME : 500, // Only used in node

		CANVASCOLOR : "eaecef",
    CELLCOLOR: ['9c9c9c'],
		zoom : 3 // zoom in on canvas with this factor.
	}
}
