from pygame.constants import K_DOWN
from pygame_player import PyGamePlayer


class AI_Player(PyGamePlayer):
    def __init__(self, force_game_fps=10, run_real_time=False):
        """
        Example class for playing Pong
        """
        super(AI_Player, self).__init__(force_game_fps=force_game_fps, run_real_time=run_real_time)
        self.last_bar1_score = 0.0
        # self.last_bar2_score = 0.0

    def get_keys_pressed(self, screen_array, feedback, terminal, turn):
        # TODO: put an actual learning agent here
        return [K_DOWN]
        # raise NotImplementedError("Please override this method")

    def get_feedback(self):
        # import must be done here because otherwise importing would cause the game to start playing
        # from pong import bar1_score, bar2_score
        from simple import bar_score, terminal_key, turn

        # score_change = bar_score - self.last_bar1_score
        # self.last_bar1_score = bar_score
        # score_change = bar_score - self.last_bar1_score
        # get the difference in score between this and the last run
        # score_change = (bar1_score - self.last_bar1_score) - (bar2_score - self.last_bar2_score)
        # self.last_bar1_score = bar1_score
        # self.last_bar2_score = bar2_score

        return float(bar_score), terminal_key, turn

    def start(self):
        super(AI_Player, self).start()

        import simple

if __name__ == '__main__':
    player = AI_Player()
    player.start()
