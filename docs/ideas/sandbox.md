Evolution Sandbox

- N Species of organisms
- Can be carnivore or herbivore
- each has a pre-determined size
- Food randomly generates
- food / organisms are colored circles (Hue, size)
- Vision is cone based ray tracing
- returns distance / color of each ray
- vision has length and arc (same number of test rays)
- moving takes energy
- movement is forward / back /turn left / turn right
- carnivore hits herbivore -> eats it
- herbivore hits food -> eats it

- energy -> 0 = dead
- food gives smaller increments faster and faster with age
- energy has passive drain rate

- spawn at random location

- fitness = total time alive + closest distance to food (to fuel early generations. max score from this is same energy it would have gotten from eating)

- Each organism has a NEAT recurrent network (CPPN?)
- inputs: 
    - vision rays
    - energy ratio
- Later inputs once proto working: 
    - random input
    - sin input
    - sin2 input
    - sin3 input
    - memory1
    - memory2
    - memory3

- physical genes f32
- color
- sin1,2,3 phase and freq
- memory phase

