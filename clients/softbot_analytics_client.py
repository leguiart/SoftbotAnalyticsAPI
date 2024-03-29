
import os
from clients.client import GenericHttpClient


class SoftbotAnalyticsClient(GenericHttpClient):
    def __init__(self, base_url):
        self.base_url = base_url
        super().__init__(connectTimeout=3600., readTimeout=14000.)
    
    def kde_distributions(self, mode, indicators, statistics, experiment_names, **kwargs):
        kde_base_url = os.path.join(self.base_url, 'IndicatorKdePlots')
        return super().generic_get(kde_base_url, 'mode', mode, indicators=indicators, statistics=statistics, experiments=experiment_names, **kwargs)
    
    def boxplots(self, mode, indicators, statistics, experiment_names, **kwargs):
        kde_base_url = os.path.join(self.base_url, 'IndicatorBoxPlots')
        return super().generic_get(kde_base_url, 'mode', mode, indicators=indicators, statistics=statistics, experiments=experiment_names, **kwargs)
    
    def violinplots(self, mode, indicators, statistics, experiment_names, **kwargs):
        kde_base_url = os.path.join(self.base_url, 'IndicatorViolinPlots')
        return super().generic_get(kde_base_url, 'mode', mode, indicators=indicators, statistics=statistics, experiments=experiment_names, **kwargs)

    def bs_convergence(self, indicators, statistics, experiment_names, n_boot = 'default', **kwargs):
        bs_base_url = os.path.join(self.base_url, 'IndicatorBsConvergencePlots')
        return super().generic_get(bs_base_url, 'n_boot', n_boot, indicators=indicators, statistics=statistics, experiments=experiment_names, **kwargs)

    def choose_winner(self, statistic, indicators, experiment_names, **kwargs):
        winner_base_url = os.path.join(self.base_url, 'ChooseWinner')
        return super().generic_get(winner_base_url, 'statistic', statistic, indicators=indicators, experiments=experiment_names, **kwargs)

    def joint_kde(self, mode, indicator1, indicator2, statistics, experiment_names, **kwargs):
        _base_url = os.path.join(self.base_url, 'IndicatorJointKdePlot')
        return super().generic_get(_base_url, 'indicator1', indicator1, 'indicator2', indicator2, 'mode', mode, statistics=statistics, experiments=experiment_names, **kwargs)

    def pairplot_kde(self, mode, indicators, statistics, experiment_names, **kwargs):
        _base_url = os.path.join(self.base_url, 'IndicatorPairPlots')
        return super().generic_get(_base_url, 'mode', mode, indicators=indicators, statistics=statistics, experiments=experiment_names, **kwargs)
    
    def structured_archive_plot(self, archive, indicator, statistic, experiment_names, **kwargs):
        _base_url = os.path.join(self.base_url, 'StructuredArchivePlots')
        return super().generic_get(_base_url, 'archive', archive, 'indicator', indicator, 'statistic', statistic, experiments=experiment_names, **kwargs)

