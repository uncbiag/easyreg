reproduce_paper_result=False
# We fix coordinate definition in the identity map, which only affect the learning method in easyreg.
# From learning perspective, this fix should not affect the network convergence.
# We add this flag here to ensure the paper results to be reproducible
# We suggest to set it false in other cases.
