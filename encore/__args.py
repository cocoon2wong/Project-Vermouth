"""
@Author: Conghao Wong
@Date: 2025-12-02 11:09:18
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-05 19:53:24
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class EncoreArgs(EmptyArgs):

    # ---------------------
    # MARK: - Model Configs
    # ---------------------

    @property
    def Kg(self) -> int:
        """
        The number of generations when making predictions.
        It is also the channels of the generating kernel in the proposed
        reverberation transform.
        """
        return self._arg('Kg', 20, argtype=STATIC,
                         desc_in_model_summary='Generating channels')

    @property
    def partitions(self) -> int:
        """
        The number of partitions when computing the angle-based feature.
        It is only used when modeling social interactions.
        """
        return self._arg('partitions', -1, argtype=STATIC,
                         desc_in_model_summary='Number of Angle-based Partitions')

    @property
    def T(self) -> str:
        """
        Transform type used to compute trajectory spectrums.

        It could be:
        - `none`: no transformations;
        - `haar`: haar wavelet transform;
        - `db2`: DB2 wavelet transform.
        """
        return self._arg('T', 'haar', argtype=STATIC, short_name='T',
                         desc_in_model_summary='Transform type')

    @property
    def ego_t_h(self) -> int:
        """
        Input length of the ego predicotr.
        """
        return self._arg('ego_t_h', -1, STATIC,
                         desc_in_model_summary=('Ego predictor',
                                                'Input length'))

    @property
    def ego_t_f(self) -> int:
        """
        Output length of the ego predicotr.
        """
        return self._arg('ego_t_f', -1, STATIC,
                         desc_in_model_summary=('Ego predictor',
                                                'Output length'))

    @property
    def ego_loss_rate(self) -> float:
        """
        Loss weight of the EgoLoss when training.
        """
        return self._arg('ego_loss_rate', 0.6, STATIC,
                         desc_in_model_summary=('Ego predictor',
                                                'EgoLoss Weight'))

    @property
    def ego_capacity(self) -> int:
        """
        The number of neighbors (`N`) to be "well-forecasted" by the ego
        predictor. When there are more numbers of neighbors than this value
        in the scene, the ego predictor will choose the most-`N` closed
        neighbors in relation to the ego agent to run the full-size prediction,
        while other neighbors will be forecasted using a simple linear
        predictor.

        **Ablation Settings:**
        Note that the full-size Transformer-based ego predictor will be
        constructed and used for prediction only when `N > 0`. A linear
        predictor will be used for all neighbors when `N` is set to `0`.
        """
        return self._arg('ego_capacity', -1, DYNAMIC,
                         desc_in_model_summary=('Ego predictor',
                                                'Capacity'))

    @property
    def insights(self) -> int:
        """
        The number of "insights" (`I`) in the ego predictor. The full-size ego
        predictor will forecast `I` short-term trajectories for each neighbor
        within its capacity.
        """
        return self._arg('insights', 5, STATIC,
                         desc_in_model_summary=('Ego predictor',
                                                'Number of Insights'))

    @property
    def encode_agent_types(self) -> int:
        """
        Choose whether to encode the type name of each agent.
        It is mainly used in multi-type-agent prediction scenes, providing
        a unique type-coding for each type of agents when encoding their
        trajectories.
        """
        return self._arg('encode_agent_types', 0, argtype=STATIC)

    # ---------------------
    # MARK: - Ablation Args
    # ---------------------
    @property
    def use_linear(self) -> int:
        """
        **Ablation Settings:**
        (bool) Choose whether to use the linear prediction as the base of all
        other predictions.
        """
        return self._arg('use_linear', 1, STATIC,
                         desc_in_model_summary=('Ablation Settings',
                                                'Use linear prediction base'))

    @property
    def use_intention_predictor(self) -> int:
        """
        **Ablation Settings:**
        (bool) Choose whether to use the intention prediction as one of the model
        predictions.
        """
        return self._arg('use_intention_predictor', 1, STATIC,
                         desc_in_model_summary=('Ablation Settings',
                                                'Use intention predictor'))

    @property
    def use_social_predictor(self) -> int:
        """
        **Ablation Settings:**
        (bool) Choose whether to use the social prediction as one of the model
        predictions.
        """
        return self._arg('use_social_predictor', 1, STATIC,
                         desc_in_model_summary=('Ablation Settings',
                                                'Use social predictor'))

    # ----------------
    # MARK: - Vis Args
    # ----------------

    @property
    def vis_ego_predictor(self) -> int:
        """
        Choose whether to visualize trajectories forecasted by the ego
        predictior.
        It accepts three values:

        - `0`: Do nothing;
        - `1`: Visualize ego predictor's all predictions;
        - `2`: Visualize ego predictor's mean predicton for each neighbor.

        NOTE that this arg only works in the *Playground* mode, or the program
        will be killed immediately.
        """
        return self._arg('vis_ego_predictor', 0, argtype=TEMPORARY)

    def _init_all_args(self):
        super()._init_all_args()

        if self.T == 'fft':
            self.log(f'Transform `{self.T}` is not supported!',
                     level='error', raiseError=ValueError)

        if -1 in [self.ego_t_h, self.ego_t_f]:
            self.log('Please specify input or output lengths of the ego ' +
                     'predictor! Currently received: ' +
                     f'(t_h, t_f) = ({self.ego_t_h}, {self.ego_t_f}).',
                     level='error', raiseError=ValueError)

        if self.use_social_predictor + self.use_intention_predictor == 0:
            self.log('`use_social_predictor` and `use_intention_predictor` ' +
                     'can not be disabled simultaneously!',
                     level='error', raiseError=ValueError)

        if self.partitions <= 0:
            self.log(f'Illegal partition settings ({self.partitions})! ' +
                     'Please add the arg `--partitions` to set the number of ' +
                     'angle-based partitions.',
                     level='error', raiseError=ValueError)

        if ((self.vis_ego_predictor)
                and (self._terminal_args is not None)
                and ('playground' not in ''.join(self._terminal_args))):
            self.log('Arg `vis_ego_predictor` can be only used in the ' +
                     'playground mode!',
                     level='error', raiseError=ValueError)
