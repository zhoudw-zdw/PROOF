def get_model(model_name, args):
    name = model_name.lower()
    if name=="proof":
        from models.proof import Learner
        return Learner(args)
    
    else:
        assert 0
