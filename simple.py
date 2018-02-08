#!/usr/bin/env python
# coding: utf-8
import math
import sys
import random
from pygame.locals import *
# import .base
from ple.games.base.pygamewrapper import PyGameWrapper
import os
import pygame
from pygame.constants import K_w, K_s
# from utils.vec2d import vec2d

screen_x = 128
screen_y = 128
angle = 0
SCR_RECT = Rect(0, 0, screen_x, screen_y)
bar_score = 0
FINISH_SCORE = 100
perepisode = 100
terminal_key = False
keynum = 0
stock = 0
times = 0
last_time = 0
final = False


# angle = 360
def load_image(filename, colorkey=None):
    """画像をロードして画像と矩形を返す"""
    filename = os.path.join("image", filename)
    try:
        image = pygame.image.load(filename)
    except pygame.error, message:
        print "Cannot load image:", filename
        raise SystemExit, message
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self)
        global bar_score, terminal_key
        self.level = level
        self.image = pygame.Surface((16, 16), 0, 32)
        if level == 0:
            self.rect = self.image.fill((229, 87, 0))
        elif level == 1:
            self.rect = self.image.fill((255, 113, 25))
        elif level == 2:
            self.rect = self.image.fill((255, 144, 76))
        else:
            self.rect = self.image.fill((255, 168, 114))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self)
        self.level = level
        self.depth = 51 - (self.level + 1) * 51
        self.image = pygame.Surface((16, 16), 0, 32)
        self.rect = self.image.fill((89, 34, 0))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""

    def __init__(self, bricks, holes, angle):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image("paddle4.png")
        colorkey = self.image.get_at((0, 2))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self._image = self.image
        # self.forward = 16
        # self.back = 16
        self.angle = angle
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.alls = all
        self.rect.center = (72, 120)
        self.x, self.y = self.rect.center

    def rotate(self):
        center = self.rect.center
        self.image = pygame.transform.rotate(self._image, self.angle)
        self.rect = self.image.get_rect(center=center)

    def draw(self, screen):
        self.rect.clamp_ip(SCR_RECT)
        screen.blit(self.image, (self.rect.left, self.rect.top))
        # print self.rect.left, self.rect.top


class Agent(PyGameWrapper):
    """
    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    """

    def __init__(self, width=128, height=128):
        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }
        self.keyup = False
        self.keydown = False
        self.keyright = False
        self.keyleft = False
        self.key = 0
        self.keynum = 0
        self.perepisode = 100
        PyGameWrapper.__init__(self, width, height, actions=actions)

    def _handle_player_events(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                self.key = event.key
                self.keynum += 1
                # 重機の操縦(初代+旋回simple)
                vx = vy = 16
                if self.key == self.actions["up"]:
                    self.keyup = True
                    self.player.rect.move_ip(0, -vy)
                    self.player.angle = 0
                elif self.key == self.actions["down"]:
                    self.keydown = True
                    self.player.rect.move_ip(0, vy)
                    self.player.angle = 180
                elif self.key == self.actions["right"]:
                    self.keyright = True
                    self.player.rect.move_ip(vx, 0)
                    self.player.angle = 270
                elif self.key == self.actions["left"]:
                    self.keyleft = True
                    self.player.rect.move_ip(-vx, 0)
                    self.player.angle = 90

    def collide(self):
        a = []
        num = 0
        num2 = 0
        bricks_collided = pygame.sprite.spritecollide(self.player, self.bricks, False)
        holes_collided = pygame.sprite.spritecollide(self.player, self.holes, False)
        if holes_collided:
            if self.keyup == True:
                self.player.rect.top += 16
            elif self.keydown == True:
                self.player.rect.top -= 16
            elif self.keyright == True:
                self.player.rect.left -= 16
            elif self.keyleft == True:
                self.player.rect.left += 16

        if bricks_collided:  # 衝突ブロックがある場合

            for brick in bricks_collided:  # 各衝突ブロックに対して
                if self.keyup == True:
                    brick.rect.top -= 16
                    self.score += 1
                if self.keydown == True:
                    brick.rect.top += 16
                    self.score -= 1
                if self.keyright == True:
                    brick.rect.left += 16
                if self.keyleft == True:
                    brick.rect.left -= 16


                    ####

                a = pygame.sprite.spritecollide(brick, self.bricks, False)

                # 四辺の判定
                if brick.rect.left < 0:
                    brick.rect.left = 0
                    self.player.rect.left = 16
                    # bar_score -= 5
                if brick.rect.top < 0:
                    brick.rect.top = 0
                    self.player.rect.top = 16
                    self.score -= 1
                if brick.rect.right > screen_x:
                    brick.rect.right = screen_x
                    self.player.rect.left = screen_x - self.player.rect.width - 16
                    # bar_score -= 5
                if brick.rect.bottom > screen_y:
                    brick.rect.bottom = screen_y
                    self.player.rect.top = screen_y - self.player.rect.height - 16
                    #  bar_score -= 5

                # brickとholeの衝突判定
                bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                if bricks_collided2:
                    pygame.sprite.Group.remove(self.bricks, brick)
                    # pygame.sprite.Group.remove(self.alls, brick)
                    # 衝突しているbrickとholeをgroupから削除
                    for hole in bricks_collided2:
                        pygame.sprite.Group.remove(self.holes, hole)
                        # pygame.sprite.Group.remove(self.alls, hole)
                        self.score += 10
                        break

        # brick同士の衝突(重ねられなくする処理)
        if len(a) > 4:
            for brick in bricks_collided:
                num2 += 1
                if self.keyup == True:
                    brick.rect.top += 16
                    # print "up"
                if self.keydown == True:
                    brick.rect.top -= 16
                    # print "down"
                if self.keyright == True:
                    brick.rect.left -= 16
                    # print "right"
                if self.keyleft == True:
                    brick.rect.left += 16
                    # print "left"
                if len(bricks_collided) - num2 == 3:
                    break
        # 土砂の高さをdepthに変換
        for brick in self.bricks:
            old_x = brick.rect.left / 16
            old_y = brick.rect.top / 16
            pygame.sprite.Group.remove(self.bricks, brick)
            # pygame.sprite.Group.remove(self.alls, brick)
            a = pygame.sprite.spritecollide(brick, self.bricks, False)
            self.bricks.add(Brick(old_x, old_y, len(a)))

        # 穴の深さをdepthに変換
        for hole in self.holes:
            old_x = hole.rect.left / 16
            old_y = hole.rect.top / 16
            pygame.sprite.Group.remove(self.holes, hole)
            # pygame.sprite.Group.remove(self.alls, hole)
            b = pygame.sprite.spritecollide(hole, self.holes, False)
            self.holes.add(Hole(old_x, old_y, len(b)))

        self.player.rect.clamp_ip(SCR_RECT)
        # pygame.sprite.Group.remove(self.alls, self)
        # pygame.sprite.Group.add(self.alls, self)
        # 穴を全部埋めたら報酬大
        if len(self.holes) == 0:
            self.score += 20
            # print "score=",self.score
            # print "hole=",len(self.holes)
            # print bar_score

    def reset_param(self):
        self.keyup = False
        self.keydown = False
        self.keyright = False
        self.keyleft = False
        self.key = 0

    # def reset_keynum(self):
    #     self.keynum = 0

    def getActions(self):
        return self.actions.values()

    def getScore(self):
        return self.score

    def game_over(self):
        # return self.lives <= 0.0
        return len(self.holes) == 0 or self.perepisode < self.keynum

    def init(self):
        self.score = 0
        self.bricks = pygame.sprite.Group()  # 衝突判定用グループ
        self.holes = pygame.sprite.Group()
        brick_random_x = random.randint(-1, 1)
        brick_random_y = random.randint(-1, 0)
        self.bricks.add(Brick(3 + brick_random_x, 5 + brick_random_y, 2),
                        Brick(4 + brick_random_x, 5 + brick_random_y, 2),
                        Brick(4 + brick_random_x, 5 + brick_random_y, 1),
                        Brick(5 + brick_random_x, 5 + brick_random_y, 1),
                        Brick(4 + brick_random_x, 4 + brick_random_y, 1),
                        Brick(4 + brick_random_x, 6 + brick_random_y, 1))

        self.holes.add(Hole(3, 0, 0),
                       Hole(4, 0, 0),
                       Hole(2, 0, 0),
                       Hole(5, 0, 0),
                       Hole(3, 1, 0),
                       Hole(4, 1, 0))

        self.player = Paddle(self.bricks, self.holes, angle=0, )

        clock = pygame.time.Clock()

    def reset(self):
        self.init()
        self.keynum = 0

    def step(self, dt):
        self.score = 0
        self.screen.fill((140, 53, 0))
        self._handle_player_events()
        self.player.rotate()
        self.collide()
        self.reset_param()
        self.bricks.draw(self.screen)
        self.holes.draw(self.screen)
        self.player.draw(self.screen)
        # print self.score
        # print self.bricks.image
        # print len(self.holes)


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = Agent(width=128, height=128)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        if game.game_over():
            game.reset()
            # game.reset_keynum()
            print game.getScreenRGB()
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
