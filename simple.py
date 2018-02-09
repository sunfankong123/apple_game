#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import math
import sys
import random
from pygame.locals import *
# import .base
from ple.games.base.pygamewrapper import PyGameWrapper
import os
import pygame
from pygame.constants import K_w, K_s

screen_x = 128
screen_y = 192
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
    except pygame.error as message:
        print("Cannot load image:", filename)
        raise SystemExit(message)
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
        self.image = pygame.Surface((8, 8), 0, 32)
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
        self.image = pygame.Surface((8, 8), 0, 32)
        self.rect = self.image.fill((89, 34, 0))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""

    def __init__(self, bricks, holes):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image("paddle4.png")
        colorkey = self.image.get_at((0, 2))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self._image = self.image
        # self.forward = 16
        # self.back = 16
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.alls = all
        self.rect.center = (40, 186)
        self.x, self.y = self.rect.center

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

    def __init__(self, width=128, height=192):
        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }
        self.keyup = False
        self.keydown = False
        self.keyupright = False
        self.keyupleft = False
        self.key = 0
        self.keynum = 0
        self.perepisode = 500
        self.depo = 0
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
                vx = 8
                vy = 16
                if self.key == self.actions["up"]:
                    self.keyup = True
                    self.player.rect.move_ip(0, -vy)
                    self.player.angle = 0
                    break
                elif self.key == self.actions["down"]:
                    self.keydown = True
                    self.player.rect.move_ip(0, vy)
                    self.player.angle = 180
                    break
                elif self.key == self.actions["right"]:
                    self.keyupright = True
                    self.player.rect.move_ip(vx, -vy)
                    self.player.angle = 270
                    break
                elif self.key == self.actions["left"]:
                    self.keyupleft = True
                    self.player.rect.move_ip(-vx, -vy)
                    self.player.angle = 90
                    break

                # self.player.rect.clamp_ip(SCR_RECT)

    def collide(self):
        numa = 0
        numb = 0
        numc = 0
        numd = 0
        a = []
        b = []
        c = []
        d = []
        # if self.player.rect.top == 64 and self.keyupright
        # print self.player.rect.top
        holes_collided = pygame.sprite.spritecollide(self.player, self.holes, False)
        if holes_collided:
            if self.keyup == True:
                self.player.rect.top += 8
            elif self.keydown == True:
                self.player.rect.top -= 8
            elif self.keyupright == True:
                self.player.rect.left -= 8
                self.player.rect.top += 16
            elif self.keyupleft == True:
                self.player.rect.left += 8
                self.player.rect.top += 16
        holes_collided = pygame.sprite.spritecollide(self.player, self.holes, False)
        if holes_collided:
            if self.keyup == True:
                self.player.rect.top += 8
            elif self.keydown == True:
                self.player.rect.top -= 8
        #     elif self.keyupright == True:
        #         self.player.rect.top += 8
        #     elif self.keyupleft == True:
        #         self.player.rect.top += 8
        #########
        # break #
        #########
        if (holes_collided and self.keyupright) or (holes_collided and self.keyupleft):
            pass
        else:
            bricks_collided = pygame.sprite.spritecollide(self.player, self.bricks, False)
            if bricks_collided:  # 衝突ブロックがある場合
                # print "-------------------------------------------------------"
                # print "X＝", self.rect.left / 32
                # print "Y＝", self.rect.top / 32

                for brick in bricks_collided:  # 各衝突ブロックに対して
                    # print "-------------"
                    # print "X＝",brick.rect.left / 32
                    # print "Y＝",brick.rect.top / 32
                    # print "-------------"
                    # oldrect_b = copy.copy(brick.rect)
                    # print brick.rect.top
                    # print self.player.rect.top
                    # print "--"
                    if self.keyup == True:
                        self.score += 1
                        # print "aa"
                        if brick.rect.top == self.player.rect.top:
                            brick.rect.top -= 8
                        elif brick.rect.top == self.player.rect.top + 8:
                            brick.rect.top -= 16
                            collided_half = True
                    elif self.keydown == True:
                        if brick.rect.bottom == self.player.rect.bottom:
                            brick.rect.top += 8
                        else:
                            brick.rect.top += 16
                            self.score -= 1
                    elif self.keyupright == True:
                        self.score += 1
                        if self.player.rect.top == brick.rect.top:
                            brick.rect.top -= 8
                        elif self.player.rect.top == brick.rect.top - 8:
                            brick.rect.top -= 16
                            brick.rect.left += 8

                    elif self.keyupleft == True:
                        self.score += 1
                        if self.player.rect.top == brick.rect.top:
                            brick.rect.top -= 8
                        elif self.player.rect.top == brick.rect.top - 8:
                            brick.rect.top -= 16
                            brick.rect.left -= 8

                    if brick.rect.left == self.player.rect.left:
                        # a += 1
                        a = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.left == self.player.rect.left + brick.rect.width:
                        # b += 1
                        b = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.player.rect.top + brick.rect.height:
                        # c += 1
                        c = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.player.rect.top:
                        # d += 1
                        d = pygame.sprite.spritecollide(brick, self.bricks, False)

                    # print a, b, c, d
                    # print len(a), len(b), len(c), len(d)

                    # 四辺の判定
                    if brick.rect.left < 0:
                        brick.rect.left = 0
                        self.score -= 20
                        self.player.rect.left = 8
                        # bar_score -= 5
                    if brick.rect.top < 0:
                        brick.rect.top = 0
                        self.player.rect.top = 8
                        self.score -= 1
                    if brick.rect.right > screen_x:
                        brick.rect.right = screen_x
                        self.player.rect.left = screen_x - self.player.rect.width - 8
                        # bar_score -= 5
                        self.score -= 20
                    if brick.rect.bottom > screen_y:
                        brick.rect.bottom = screen_y
                        self.player.rect.top = screen_y - self.player.rect.height - 8
                        self.score -= 20
                        # bar_score -= 5

                    # brickとholeの衝突判定
                    bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                    if bricks_collided2:
                        pygame.sprite.Group.remove(self.bricks, brick)
                        # 衝突しているbrickとholeをgroupから削除
                        for hole in bricks_collided2:
                            pygame.sprite.Group.remove(self.holes, hole)
                            self.score += 15
                            break

                # 土砂シミュレーション

                if len(a) > 4:
                    for brick in bricks_collided:
                        if self.player.rect.left == brick.rect.left:
                            numa += 1
                            brick.rect.left -= 8
                            if len(a) - numa == 4:
                                break

                elif len(b) > 4:
                    for brick in bricks_collided:
                        if self.player.rect.left == brick.rect.left - brick.rect.width:
                            numb += 1
                            brick.rect.left += 8
                            if len(b) - numb == 4:
                                break

                elif len(c) > 4:
                    for brick in bricks_collided:
                        if self.player.rect.top == brick.rect.top - brick.rect.height:
                            numc += 1
                            brick.rect.top += 8
                            self.score -= 1
                            if len(c) - numc == 4:
                                break

                elif len(d) > 4:
                    for brick in bricks_collided:
                        if self.player.rect.top == brick.rect.top:
                            numd += 1
                            brick.rect.top -= 8
                            self.score += 1
                            if brick.rect.top < 0:
                                brick.rect.top = 0
                                self.score -= 1
                            if len(d) - numd == 4:
                                break

                """
            # brick同士の衝突(取り残される処理)
            elif len(bricks_collided) > 3:
                for brick in bricks_collided:
                    num += 1
                    if keyup == True:
                        brick.rect.top += 8
                    if keydown == True:
                        brick.rect.top -= 8
                    if keyright == True:
                        brick.rect.left -= 8
                    if keyleft == True:
                        brick.rect.left += 8
                    if len(bricks_collided) - num == 3:
                        break
            """
            # 土砂の高さをdepthに変換
            for brick in self.bricks:
                if brick.rect.top <= 0:
                    self.score -= 20
                    pygame.sprite.Group.remove(self.bricks, brick)
                #     topend = True
                else:
                    old_x = brick.rect.left / 8
                    old_y = brick.rect.top / 8
                    pygame.sprite.Group.remove(self.bricks, brick)
                    a = pygame.sprite.spritecollide(brick, self.bricks, False)
                    self.bricks.add(Brick(old_x, old_y, len(a)))

            # 穴の深さをdepthに変換
            # for hole in self.holes:
            #     old_x = hole.rect.left / 8
            #     old_y = hole.rect.top / 8
            #     pygame.sprite.Group.remove(self.holes, hole)
            #     b = pygame.sprite.spritecollide(hole, self.holes, False)
            #     self.holes.add(Hole(old_x, old_y, len(b)))

            self.player.rect.clamp_ip(SCR_RECT)
            if len(self.bricks) <= 5:
                self.depo += 2
                a = self.depo % 8
                # b = depo // 13
                brick_random_x = a
                brickX = 4
                brickY = 14
                self.bricks.add(
                    Brick(brickX + brick_random_x, brickY, 0),
                    Brick(brickX + 1 + brick_random_x, brickY, 0),
                    Brick(brickX + 2 + brick_random_x, brickY, 0),
                    Brick(brickX + brick_random_x, brickY + 1, 0),
                    Brick(brickX + 1 + brick_random_x, brickY + 1, 1),
                    Brick(brickX + 1 + brick_random_x, brickY + 1, 1),
                    Brick(brickX + 2 + brick_random_x, brickY + 1, 0),
                    Brick(brickX + brick_random_x, brickY + 2, 0),
                    Brick(brickX + 1 + brick_random_x, brickY + 2, 0),
                    Brick(brickX + 2 + brick_random_x, brickY + 2, 0))
            # 穴を全部埋めたら報酬大
            if len(self.holes) == 16:
                self.score += 50

    def reset_param(self):
        self.keyup = False
        self.keydown = False
        self.keyupright = False
        self.keyupleft = False
        self.key = 0

    def getActions(self):
        return self.actions.values()

    def getScore(self):
        return self.score

    def game_over(self):
        # return self.lives <= 0.0
        return len(self.holes) == 16 or self.perepisode <= self.keynum

    def init(self):
        self.depo = 0
        self.score = 0
        self.bricks = pygame.sprite.Group()  # 衝突判定用グループ
        self.holes = pygame.sprite.Group()
        brickX = 4
        brickY = 14

        self.bricks.add(Brick(brickX, brickY, 0),
                        Brick(brickX + 1, brickY, 0),
                        Brick(brickX + 2, brickY, 0),
                        Brick(brickX, brickY + 1, 0),
                        Brick(brickX + 1, brickY + 1, 1),
                        Brick(brickX + 1, brickY + 1, 1),
                        Brick(brickX + 2, brickY + 1, 0),
                        Brick(brickX, brickY + 2, 0),
                        Brick(brickX + 1, brickY + 2, 0),
                        Brick(brickX + 2, brickY + 2, 0))
        for j in range(-1, 8, 1):
            for i in range(0, 16, 1):
                self.holes.add(Hole(i, j, 0))

        self.player = Paddle(self.bricks, self.holes)

        clock = pygame.time.Clock()

    def reset(self):
        self.keynum = 0
        self.init()

    def step(self, dt):
        self.score = 0
        self.screen.fill((140, 53, 0))
        self._handle_player_events()
        self.collide()
        self.reset_param()
        self.bricks.draw(self.screen)
        self.holes.draw(self.screen)
        self.player.draw(self.screen)
        # print self.player.rect.bottom
        # print self.score
        # print self.bricks.image
        # print len(self.holes)


if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = Agent()
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        if game.game_over():
            game.reset()
            # print game.getScreenRGB()
        dt = game.clock.tick_busy_loop(30)
        game.step(dt)
        pygame.display.update()
