"""Published PRETRAIN recipe and selected transfer recipe only."""
PRETRAIN = {'initialization': 'scratch', 'model': 'd3', 'objective': 'd3', 'lr': 0.001}
SELECTED = {'initialization': 'source_student', 'model': 'd3', 'objective': 'd3', 'lr': 0.0002}
WINNER_ID = 'TS2'
