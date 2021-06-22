import typing

# Given: {D_0, y_0}, where D_0 \cup D (for example random 10% of stanford dogs)
# L = D/D_0
# for i in range(N):
#    1) train model on D_0 (could be not needed: random case)
#    2) let {D', y_hat} be chosen examples from L using model
#    3) let {D', y*} be the labels using oracle strategy
#    4) D_0 <- {D_0, y_0} and {D', y*} according to combine strategy
#    5) L = D/D_0

# 1,2) filter_strategy
# 3) oracle_strategy
# 4) combine_strategy

# {D_0, y_0} -> confidence level 1
# {D', y} -> confidence level 0 <= l <= 1

# NOTE: most important part is index managing for databases

# 1) database polling
# 

# 2) image selection
# 2.1) model training (if needed)
# 2.2) image selection

class image_selector:
    def select(images):
        pass
        # 1) pick random
        # 2) pick images whose model output is above a threshold
        # etc.


# strategy needs to be broken down or clarified/abstracted? 
class train_select_strategy:
    def __init__(self, strategy:image_selector) -> None:
        pass

    def iterate(self, train_images):
        self.train(train_images)
        self.select(train_images)

    def train(self, train_images):
        strategy.do_train(train_images)

    def select(self, images):
        strategy.do_select(images)


def start_loop(N:int, filter:filter_strategy, ):
    '''
    N, number of iterations
    filter, 
    '''
    for i in range(N):
        pass
    
