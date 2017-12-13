import gaussfitter

input_parameters = [0, 1, 25, 25, 3, 6, 30]
print("Input parameters are: {0}".format(input_parameters))
example_gaussian = gaussfitter.twodgaussian(input_parameters, shape=[50,50],)

moments = gaussfitter.moments(example_gaussian, circle=False, rotate=True, vheight=True)

print("These are the input guesses computed using 'moments':")
print(moments)

print("This is from the gaussian fit; it works well:")
print(gaussfitter.gaussfit(example_gaussian))
