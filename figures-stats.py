from pathlib import Path

from libs.figures import figure_1_univariate_referencing
from libs.figures import figure_2_multivariate_referencing

path = Path(r'output\\20260728132320')

figure_1_univariate_referencing.make(path)
figure_2_multivariate_referencing.make(path)

print('Done')