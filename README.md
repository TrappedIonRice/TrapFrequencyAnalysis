To ensure your machine's python is compatible do the following: (if you dont have python at all download it first, preferably version 3.11.9)

In your cmd prompt run "where py" ((this checks if you have the windows python launcher)) 
  If you get a valid path, then: 
    Run "py -3.11 - V" 
      if this outputs Python 3.11.__ 
        your good! 
      if not: 
        run "winget install -e --id Python.Python.3.11" 
  If you do not get a valid path, then:
     run "winget install -e --id Python.Python.3.11" 

  To check, close and reopen cmd prompt and run "where py" then "py -3.11 -V" for verification.





General Data Flow:
1. Extract Discrete data from a Comsol sim for each Electrode
2. Combine all electrodes raw data into a single data table
3. Determine input dependent values
4. From here call desired fucntionality such as...
  - find min of potential
  - fit potential at a given point (such as min)
  - Graph desired slice of data (raw or procesed)
  - And more

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Current inputs:
  - An instance of the Elctrode_Vars class
    - This is a class in which [FreqAmp, Freq, constant_offset, phase] are stored for each electrode
  - Other vlaues in the constants file, these are not currently accesable as varibles for each run as they are determined physicaly

Curent (quantitativly relavant) outputs:
  - Frequency in the x,y,z directions at a given point
  - Principla frequency and their directions at a given point
  - Fitting parameters for totalV at a given point

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Notes on each step:

Extracting raw data:
  - Give the electorde of intrest a potential of 1 and ground all others, export a 3dpoint grid for this simualtion
  - Repeat for all electrodes, (ensure same grid is exported)

Combine data:
  - Curently each electrod has its own data file and then there is also a combined simualtion data file
  - In the inital extraction ensure all values are in SI (ie: mm --> m)
  - The combined data frame will have for each point (x,y,z) the columns (values):
    - For each electrode (DCq) --> columns DCq_Ex, DCq_Ey, DCq_Ez, DCq_V
    - and a column holding the total potential/pseudopotential (this is inpout dependent)
   
Determine the input dependent values:
  - As of right now this is just totalV (as given by the pseudopotential eq + regular V)
  - will change if inputs expanded (IE: if number of ions becomes a variable)

Call desired fucntionality:
  - As of now all fucntionality is a fucntion wihtin the simualtion class, or in the experiment_funcs file.
  - There are docstrings for each fucntion

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Known problems:
  - The fitting fucntion when using deg2 contains error that grows with the size of the neighborhood fitted
    this is largely mitigated by using deg4 fit, however with higher dim fits the point at which the deg4 terms "kick in"
    is delayed, thus I conjeture it is best practuce to fit a small neighborhood or a maximal one as middle grounds give rise to uncertintiy.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Future:
  - Find a reasonable way to determien the output space of this code, and then find a way to take in a desired output and return the corosponding inputs
  - Work in multi ion dynamics
  - Get a less simplified and mroe refined COMSOL sim

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

Last updated README: 4/3/2025, Evan Miller
