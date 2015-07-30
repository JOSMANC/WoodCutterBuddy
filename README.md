#WoodCutterBuddy
##Purpose
A python application to solve the optimal amount of raw material needed for a carpentry or cutting project by solving the *cutting stock* problem.  It considers the amount of materials required and reports the minimal number of raw materials (8 ft 2x4) required and where the raw material should be cut.

The program takes the wood sizes and how many of that size are required for the project and then outputs a text solution and a full schematic.  A small cutting error is considered in all cutting patterns.   The [api](https://github.com/JOSMANC/WoodCutterBuddy/blob/master/WoodCutterBuddy_api.py) file outputs instructions for how to use the program. 

requires packages `numpy`, `matplotlib` and `collections`

![](https://github.com/JOSMANC/WoodCutterBuddy/blob/master/image/woodbuddyschematic.png)

##Algorithm
Details to come, but in the mean time please consult https://en.wikipedia.org/wiki/Cutting_stock_problem

##Future Features
- Deploy as online app
- Flag non-optimal solution events
- Account for non-optimal solution events
