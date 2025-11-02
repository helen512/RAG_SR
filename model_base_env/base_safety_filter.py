'''Base class for safety filter.'''

from abc import abstractmethod




class BaseSafetyFilter():
    '''Template for safety filter, implement the following methods as needed.'''

    @abstractmethod
    def certify_action(self,
                       current_state,
                       uncertified_action,
                       info=None,
                       ):
        '''Determines a safe action from the current state and proposed action.

        Args:
            current_state (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            info (dict): The info at this timestep.

        Returns:
            certified_action (ndarray): The certified action.
            success (bool): Whether the safety filtering was successful or not.
        '''
        raise NotImplementedError

    def select_action(self, obs, info=None):
        raise NotImplementedError('[ERROR] select_action is not and will not be implemented for safety filters.')

    def get_prior(self, env, prior_info={}):
        '''Fetch the prior model from the env for the controller.

        Note there's a default env.symbolic when each each env is created.
        To make a different prior model, do the following when initializing a ctrl::

            self.env = env_func()
            self.model = self.get_prior(self.env)

        Besides the env config `base.yaml` and ctrl config `mpc.yaml`,
        you can define an additional prior config `prior.yaml` that looks like::

            algo_config:
                prior_info:
                    prior_prop:
                        M: 0.03
                        Iyy: 0.00003
                    randomize_prior_prop: False
                    prior_prop_rand_info: {}

        and to ensure the resulting config.algo_config contains both the params
        from ctrl config and prior config, chain them to the --overrides like:

            python experiment.py --algo mpc --task quadrotor --overrides base.yaml mpc.yaml prior.yaml ...

        Also note we look for prior_info from the incoming function arg first, then the ctrl itself.
        this allows changing the prior model during learning by calling::

            new_model = self.get_prior(same_env, new_prior_info)

        Alternatively, you can overwrite this method and use your own format for prior_info
        to customize how you get/change the prior model for your controller.

        Args:
            env (BenchmarkEnv): the environment to fetch prior model from.
            prior_info (dict): specifies the prior properties or other things to
                overwrite the default prior model in the env.

        Returns:
            SymbolicModel: CasAdi prior model.
        '''
        if not prior_info:
            prior_info = getattr(self, 'prior_info', {})
        prior_prop = prior_info.get('prior_prop', {})

        # randomize prior prop, similar to randomizing the inertial_prop in BenchmarkEnv
        # this can simulate the estimation errors in the prior model
        
        # randomize_prior_prop = prior_info.get('randomize_prior_prop', False)
        # prior_prop_rand_info = prior_info.get('prior_prop_rand_info', {})
        # if randomize_prior_prop and prior_prop_rand_info:
        #     # check keys, this is due to the current implementation of BenchmarkEnv._randomize_values_by_info()
        #     for k in prior_prop_rand_info:
        #         assert k in prior_prop, 'A prior param to randomize does not have a base value in prior_prop.'
        #     prior_prop = env._randomize_values_by_info(prior_prop, prior_prop_rand_info)

        # Note we only reset the symbolic model when prior_prop is nonempty
        if prior_prop:
            env._setup_symbolic(prior_prop=prior_prop)

        # Note this ensures the env can still access the prior model,
        # which is used to get quadratic costs in env.step()
        prior_model = env.symbolic
        return prior_model
