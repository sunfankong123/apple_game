#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os
import random
import copy
import math
from math import *

screen_x = 128
screen_y = 192
SCR_RECT = Rect(0, 0, screen_x, screen_y)
bar_score = 0
FINISH_SCORE = 100
perepisode = 500
terminal_key = False
keynum = 0
depo = 0
times = 0
last_time = 0
final = False
topend = False
angle = 360
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (40, 40)


# stepsfile = open("stepsdata/stepsdata.txt", "a")


def main():
    pygame.init()
    screen = pygame.display.set_mode(SCR_RECT.size, 0, 32)
    # pygame.display.set_caption(u"Breakout 06 スコアの表示")

    # スプライトグループを作成して登録
    all = pygame.sprite.OrderedUpdates()  # 描画用グループ
    bricks = pygame.sprite.Group()  # 衝突判定用グループ
    holes = pygame.sprite.Group()
    Brick.containers = all, bricks
    Hole.containers = all, holes
    Paddle.containers = all

    for j in range(0, 8, 1):
        for i in range(0, 16, 1):
            Hole(i, j, 0)

    # Hole(4, 0, 0)
    # Hole(5, 0, 0)
    # Hole(6, 0, 0)
    # Hole(7, 0, 0)
    # Hole(8, 0, 0)
    # Hole(9, 0, 0)
    #
    # Hole(10, 0, 0)
    # Hole(11, 0, 0)
    # Hole(12, 0, 0)
    # Hole(3, 0, 0)


    brickX = 4
    brickY = 14
    Brick(brickX , brickY , 0)
    Brick(brickX + 1 , brickY , 0)
    Brick(brickX + 2 , brickY , 0)
    Brick(brickX , brickY + 1 , 0)
    Brick(brickX + 1 , brickY + 1 , 1)
    Brick(brickX + 1 , brickY + 1 , 1)
    Brick(brickX + 2 , brickY + 1 , 0)
    Brick(brickX , brickY + 2 , 0)
    Brick(brickX + 1, brickY + 2 , 0)
    Brick(brickX + 2 , brickY + 2 , 0)

    # パドル
    paddle = Paddle(bricks, holes, all)

    clock = pygame.time.Clock()

    global perepisode, terminal_key, times, last_time, bar_score, final, topend
    terminal_key = False

    # current_time = int(times / 100000)
    # if current_time != last_time and perepisode > 120:
    #    last_time = current_time
    #    perepisode -= 10
    while len(holes) > 32 and keynum < perepisode and final is False and topend == False:
        clock.tick(60)
        screen.fill((140, 53, 0))
        all.update()
        all.draw(screen)
        pygame.display.update()
    terminal_key = True
    # stepsfile.write(str(keynum) + "\n")
    # print "The agent have moved %s steps" % keynum


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""

    def __init__(self, bricks, holes, all):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle4.png")
        colorkey = self.image.get_at((0, 2))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self._image = self.image
        self.forward = 8
        self.back = 8
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.alls = all
        self.rect.center = (40, 186)
        self.x, self.y = self.rect.center

    def update(self):
        # self.rect.clamp_ip(SCR_RECT)  # SCR_RECT内でしか移動できなくなる
        # turn_speed = 90
        self._rect = Rect(self.rect)
        self._rect.center = self.x, self.y
        # self.rotate()
        # turn_speed = 90
        # pygame.key.get_pressed()
        global keynum, depo
        global times
        global bar_score, topend
        # oldrect = copy.copy(self.rect)
        # num = 0

        numa = 0
        numb = 0
        numc = 0
        numd = 0
        a = []
        b = []
        c = []
        d = []
        keyup = False
        keydown = False
        keyupright = False
        keyupleft = False
        keydownleft = False
        keydownright = False

        collided_half = False
        """
        a = 0
        b = 0
        c = 0
        d = 0

        """
        # すべてのブロックに対して
        # 一回の移動量
        vx = 8
        vy = 16
        for event in pygame.event.get():  # User did something
            if event.type == KEYDOWN:
                # print bar_score
                bar_score = 0
                keynum += 1
                times += 1

                # 重機の操縦(初代+旋回simple)
                if event.key == K_UP:
                    keyup = True
                    self.rect.move_ip(0, -vy)
                elif event.key == K_DOWN:
                    keydown = True
                    self.rect.move_ip(0, vy)
                elif event.key == K_RIGHT:
                    keyupright = True
                    self.rect.move_ip(vx, -vy)
                elif event.key == K_LEFT:
                    keyupleft = True
                    self.rect.move_ip(-vx, -vy)
                    # if event.key == K_w:
                    #     keyup = True
                    #     self.rect.move_ip(0, -vy)
                    # elif event.key == K_s:
                    #     keydown = True
                    #     self.rect.move_ip(0, vy)
                    # elif event.key == K_d:
                    #     keyupright = True
                    #     self.rect.move_ip(vx, -vy)
                    # elif event.key == K_a:
                    #     keyupleft = True
                    #     self.rect.move_ip(-vx, -vy)
                    # elif event.key == K_z:
                    #     keydownleft = True
                    #     self.rect.move_ip(-vx, vy)
                    # elif event.key == K_c:
                    #     keydownright = True
                    #     self.rect.move_ip(vx, vy)
                    """
                if event.key == K_UP:
                    self.rect.left += sin(radians(self.angle)) * -self.forward
                    self.rect.top += cos(radians(self.angle)) * -self.forward
                elif event.key == K_DOWN:
                    self.rect.left += sin(radians(self.angle)) * self.back
                    self.rect.top += cos(radians(self.angle)) * self.back
                elif event.key == K_RIGHT:
                    self.angle -= turn_speed
                elif event.key == K_LEFT:
                    self.angle += turn_speed
                """
            self.rect.clamp_ip(SCR_RECT)
            # 重機と衝突したブロックリストを取得
            holes_collided = pygame.sprite.spritecollide(self, self.holes, False)
            if holes_collided:
                if keyup == True:
                    self.rect.top += 8
                elif keydown == True:
                    self.rect.top -= 8
                elif keyupright == True:
                    self.rect.left -= 8
                    self.rect.top += 8
                elif keyupleft == True:
                    self.rect.left += 8
                    self.rect.top += 8
            holes_collided = pygame.sprite.spritecollide(self, self.holes, False)
            if holes_collided:
                if keyup == True:
                    self.rect.top += 8
                elif keydown == True:
                    self.rect.top -= 8
                elif keyupright == True:
                    self.rect.top += 8
                elif keyupleft == True:
                    self.rect.top += 8
                break
            bricks_collided = pygame.sprite.spritecollide(self, self.bricks, False)
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
                    if event.key == K_UP:
                        bar_score += 1
                        if brick.rect.top == self.rect.top:
                            brick.rect.top -= 8
                        else:
                            brick.rect.top -= 16
                            collided_half = True
                    elif event.key == K_DOWN:
                        if brick.rect.bottom == self.rect.bottom:
                            brick.rect.top += 8
                        else:
                            brick.rect.top += 16
                        bar_score -= 1
                    elif event.key == K_RIGHT:
                        bar_score += 1
                        if self.rect.top == brick.rect.top:
                            brick.rect.top -= 8
                        else:
                            brick.rect.top -= 16
                            brick.rect.left += 8

                    elif event.key == K_LEFT:
                        bar_score += 1
                        if self.rect.top == brick.rect.top:
                            brick.rect.top -= 8
                        else:
                            brick.rect.top -= 16
                            brick.rect.left -= 8

                    # # brickとholeの衝突判定
                    # bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                    # holes_collided = pygame.sprite.spritecollide(self, self.holes, False)
                    # if holes_collided:
                    #     if keyup == True:
                    #         self.rect.top += 8
                    #     elif keydown == True:
                    #         self.rect.top -= 8
                    #     elif keyupright == True:
                    #         self.rect.left -= 8
                    #     elif keyupleft == True:
                    #         self.rect.left += 8
                    # if bricks_collided2:
                    #     pygame.sprite.Group.remove(self.bricks, brick)
                    #     pygame.sprite.Group.remove(self.alls, brick)
                    #     # 衝突しているbrickとholeをgroupから削除
                    #     for hole in bricks_collided2:
                    #         pygame.sprite.Group.remove(self.holes, hole)
                    #         pygame.sprite.Group.remove(self.alls, hole)
                    #         bar_score += 15
                    #         break

                    if brick.rect.left == self.rect.left:
                        # a += 1
                        a = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.left == self.rect.left + brick.rect.width:
                        # b += 1
                        b = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.rect.top + brick.rect.height:
                        # c += 1
                        c = pygame.sprite.spritecollide(brick, self.bricks, False)
                    elif brick.rect.top == self.rect.top:
                        # d += 1
                        d = pygame.sprite.spritecollide(brick, self.bricks, False)

                    # print a, b, c, d
                    # print len(a), len(b), len(c), len(d)


                    # a = pygame.sprite.spritecollide(brick, self.bricks, False)

                    # 四辺の判定
                    if brick.rect.left < 0:
                        brick.rect.left = 0
                        bar_score -= 20
                        self.rect.left = 8
                        # bar_score -= 5
                    if brick.rect.top < 0:
                        brick.rect.top = 0
                        self.rect.top = 8
                        bar_score -= 1
                    if brick.rect.right > screen_x:
                        brick.rect.right = screen_x
                        self.rect.left = screen_x - self.rect.width - 8
                        # bar_score -= 5
                        bar_score -= 20
                    if brick.rect.bottom > screen_y:
                        brick.rect.bottom = screen_y
                        self.rect.top = screen_y - self.rect.height - 8
                        bar_score -= 20
                        # bar_score -= 5

                    # brickとholeの衝突判定
                    bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                    if bricks_collided2:
                        pygame.sprite.Group.remove(self.bricks, brick)
                        pygame.sprite.Group.remove(self.alls, brick)
                        # 衝突しているbrickとholeをgroupから削除
                        for hole in bricks_collided2:
                            pygame.sprite.Group.remove(self.holes, hole)
                            pygame.sprite.Group.remove(self.alls, hole)
                            bar_score += 15
                            break

                # 土砂シミュレーション

                if len(a) > 4:
                    for brick in bricks_collided:
                        if self.rect.left == brick.rect.left:
                            numa += 1
                            brick.rect.left -= 8
                            if len(a) - numa == 4:
                                break

                elif len(b) > 4:
                    for brick in bricks_collided:
                        if self.rect.left == brick.rect.left - brick.rect.width:
                            numb += 1
                            brick.rect.left += 8
                            if len(b) - numb == 4:
                                break

                elif len(c) > 4:
                    for brick in bricks_collided:
                        if self.rect.top == brick.rect.top - brick.rect.height:
                            numc += 1
                            brick.rect.top += 8
                            bar_score -= 1
                            if len(c) - numc == 4:
                                break

                elif len(d) > 4:
                    for brick in bricks_collided:
                        if self.rect.top == brick.rect.top:
                            numd += 1
                            brick.rect.top -= 8
                            bar_score += 1
                            if brick.rect.top < 0:
                                brick.rect.top = 0
                                bar_score -= 1
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
                # if brick.rect.top == 0:
                #     bar_score -= 20
                #     topend = True
                old_x = brick.rect.left / 8
                old_y = brick.rect.top / 8
                pygame.sprite.Group.remove(self.bricks, brick)
                pygame.sprite.Group.remove(self.alls, brick)
                a = pygame.sprite.spritecollide(brick, self.bricks, False)
                Brick(old_x, old_y, len(a))

            # 穴の深さをdepthに変換
            for hole in self.holes:
                old_x = hole.rect.left / 8
                old_y = hole.rect.top / 8
                pygame.sprite.Group.remove(self.holes, hole)
                pygame.sprite.Group.remove(self.alls, hole)
                b = pygame.sprite.spritecollide(hole, self.holes, False)
                Hole(old_x, old_y, len(b))

            self.rect.clamp_ip(SCR_RECT)
            pygame.sprite.Group.remove(self.alls, self)
            pygame.sprite.Group.add(self.alls, self)
            if len(self.bricks) <= 5:
                depo += 2
                a = depo % 8
                # b = depo // 13
                brick_random_x = a
                brickX = 4
                brickY = 14
                Brick(brickX + brick_random_x, brickY , 0)
                Brick(brickX + 1 + brick_random_x, brickY , 0)
                Brick(brickX + 2 + brick_random_x, brickY , 0)
                Brick(brickX + brick_random_x, brickY + 1 , 0)
                Brick(brickX + 1 + brick_random_x, brickY + 1 , 1)
                Brick(brickX + 1 + brick_random_x, brickY + 1 , 1)
                Brick(brickX + 2 + brick_random_x, brickY + 1 , 0)
                Brick(brickX + brick_random_x, brickY + 2 , 0)
                Brick(brickX + 1 + brick_random_x, brickY + 2 , 0)
                Brick(brickX + 2 + brick_random_x, brickY + 2, 0)
            # 穴を全部埋めたら報酬大
            if len(self.holes) == 32:
                bar_score += 50
            # print "score=",bar_score
            # print "hole=",len(self.holes)


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
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
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.level = level
        self.image = pygame.Surface((8, 8), 0, 32)
        self.rect = self.image.fill((89, 34, 0))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


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


turn = 0
# s_turn = 0
while True:
    #    s_turn += 1
    #    if s_turn == 10:
    #        print "finish"
    #        stepsfile.write("finish\n")
    #        s_turn = 0
    #        turn += 1
    main()

    keynum = 0
    depo = 0
    final = False
    topend = False