#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os
import random
import math
from math import *

SCR_RECT = Rect(0, 0, 320, 240)
bar_score = 0
FINISH_SCORE = 100
perepisode = 2000
terminal_key = False
keynum = 0
stock = 0
times = 0
last_time = 0
final= False
angle = 360
RESIZED_SCREEN_X, RESIZED_SCREEN_Y = (40, 40)


def main():

    pygame.init()
    screen = pygame.display.set_mode(SCR_RECT.size, 0, 32)
    # pygame.display.set_caption(u"Breakout 06 スコアの表示")

    # スプライトグループを作成して登録
    all = pygame.sprite.RenderUpdates()  # 描画用グループ
    bricks = pygame.sprite.Group()       # 衝突判定用グループ
    holes=pygame.sprite.Group()
    Paddle.containers = all
    Brick.containers = all, bricks
    Hole.containers = all ,holes

    for x in range(1, 8, 1):
        for y in range(1, x, 1):
            Hole(x, 0, 1)
    for x in range(6, 13, 1):
        for y in range(1, x - 5, 1):
            Brick(x, 6, 1)

    # スコアボードを作成
    score_board = ScoreBoard()

    # パドル
    paddle = Paddle(bricks, holes, score_board, angle, all)

    clock = pygame.time.Clock()

    global perepisode, terminal_key, times, last_time
    terminal_key = False

    # current_time = int(times / 100000)
    # if current_time != last_time and perepisode > 120:
    #    last_time = current_time
    #    perepisode -= 10
    # while len(holes) > 0 and keynum < perepisode and final == False:
    while keynum < perepisode and final == False:
        clock.tick(60)
        screen.fill((0, 0, 0))
        all.update()
        all.draw(screen)
        pygame.display.update()

    terminal_key = True

#    stepsfile = open("stepsdata_networks_termina/stepsdata3600000.txt", "a")
#    stepsfile.write(str(keynum) + "\n")
#    print "The agent have moved %s steps" % keynum


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""
    def __init__(self, bricks, holes, score_board, angle, all):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle6.png")
        colorkey = self.image.get_at((0, 1))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self._image = self.image
        self.forward = 16
        self.back = 16
        self.angle = angle
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.alls = all
        self.score_board = score_board
        # self.x = SCR_RECT.bottom  # パドルは画面の一番下
        self.rect.center = (160,180)
        self.x, self.y = self.rect.center
    def rotate(self):
         center = self.rect.center
         self.image = pygame.transform.rotozoom(self._image, self.angle, 1.0)
         self.rect = self.image.get_rect(center=center)

    def update(self):
        # bricks.rect.clamp_ip(SCR_RECT)
        # self.image
        self.rect.clamp_ip(SCR_RECT)  # SCR_RECT内でしか移動できなくなる
        self._rect = Rect(self.rect)
        self._rect.center = self.x, self.y
        self.rotate()

        turn_speed = 90
        pygame.key.get_pressed()
        global keynum
        global times , final
        global bar_score, stock
        num = 0
        # すべてのブロックに対して
        # 一回の移動量
        vx = vy = 16

        for event in pygame.event.get():  # User did something
            if event.type == KEYDOWN:
                keynum += 1
                times += 1
                # 初代+旋回simple
                if event.key == K_UP:
                    keyup = True
                    self.rect.move_ip(0, -vy)
                    self.angle = 0
                elif event.key == K_DOWN:
                    keydown = True
                    self.rect.move_ip(0, vy)
                    self.angle = 180
                elif event.key == K_RIGHT:
                    keyright = True
                    self.rect.move_ip(vx, 0)
                    self.angle = 270
                elif event.key == K_LEFT:
                    keyleft = True
                    self.rect.move_ip(-vx, 0)
                    self.angle = 90
            # ボールと衝突したブロックリストを取得
            bricks_collided = pygame.sprite.spritecollide(self, self.bricks, False)
            if bricks_collided:  # 衝突ブロックがある場合
                oldrect = self.rect
                for brick in bricks_collided:  # 各衝突ブロックに対して
                    # brick.rect.clamp_ip(SCR_RECT)
                    # ボールが左から衝突
                    if oldrect.left < brick.rect.left < oldrect.right < brick.rect.right:
                        brick.rect.left += vx
                    # ボールが右から衝突
                    if brick.rect.left < oldrect.left < brick.rect.right < oldrect.right:
                        brick.rect.left -= vx
                    # ボールが上から衝突
                    if oldrect.top < brick.rect.top < oldrect.bottom < brick.rect.bottom:
                        brick.rect.top += vy
                        bar_score -= 1
                    # ボールが下から衝突
                    if brick.rect.top < oldrect.top < brick.rect.bottom < oldrect.bottom:
                        brick.rect.top -= vy
                        bar_score += 1

                    # Clipping Screen
                    if brick.rect.left < 0 and brick.rect.top < 0:
                        brick.rect.left += 16
                        brick.rect.top += 16
                    if brick.rect.left < 0 and brick.rect.bottom > 240:
                        brick.rect.left += 16
                        brick.rect.top -= 16
                    if brick.rect.right > 320 and brick.rect.top < 0:
                        brick.rect.left -= 16
                        brick.rect.top += 16
                    if brick.rect.right > 320 and brick.rect.bottom > 240:
                        brick.rect.left -= 16
                        brick.rect.top -= 16
                    # ボールが画面外に出たらリセット
                    if brick.rect.left < 0:
                        brick.rect.left = 0
                    if brick.rect.top < 0:
                        brick.rect.top = 0
                    if brick.rect.right > 320:
                        brick.rect.right = 320
                    if brick.rect.bottom > 240:
                        brick.rect.bottom = 240

                        # final = True
                        # bar_score = -1

                    bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                    if bricks_collided2:
                        pygame.sprite.Group.remove(self.bricks, brick)
                        pygame.sprite.Group.remove(self.alls, brick)
                        for hole in bricks_collided2:
                            num += 1
                            pygame.sprite.Group.remove(self.holes, hole)
                            pygame.sprite.Group.remove(self.alls, hole)
                            bar_score += 5
                            # if num == len(bricks_collided2):
                            break

            # 土砂の高さをdepthに変換
            for brick in self.bricks:
                old_x = brick.rect.left / 16
                old_y = brick.rect.top / 16
                pygame.sprite.Group.remove(self.bricks, brick)
                pygame.sprite.Group.remove(self.alls, brick)
                a = pygame.sprite.spritecollide(brick, self.bricks, False)
                Brick(old_x, old_y, len(a))
                # pygame.sprite.Group.add(self.bricks, brick)
            for hole in self.holes:
                old_x = hole.rect.left / 32
                old_y = hole.rect.top / 16
                pygame.sprite.Group.remove(self.holes, hole)
                pygame.sprite.Group.remove(self.alls, hole)
                b = pygame.sprite.spritecollide(hole, self.holes, False)
                Hole(old_x, old_y, len(b))
                # pygame.sprite.Group.add(self.bricks, brick)

            # else:
                # bar_score = 0
            if len(self.holes) == 0:
                bar_score += 10
            # print "socore=",bar_score
            # print "hole=",len(self.holes)


class Brick(pygame.sprite.Sprite):

    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.level = level
        self.depth = self.level * 25
        self.image = pygame.Surface((16, 16), 0, 32)
        self.rect = self.image.fill((255, self.depth, self.depth))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height

class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.level = level
        depth = 255 - self.level * 25
        self.image = pygame.Surface((32, 16), 0, 32)
        self.rect = self.image.fill((depth, depth, 255))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class ScoreBoard():
    """スコアボード"""
    def __init__(self):
        self.sysfont = pygame.font.SysFont(None, 17)
        # self.score = 0

#    def draw(self, screen):
#        score_img = self.sysfont.render("SCORE:" + str(bar_score), True, (255, 255, 255))
#        screen.blit(score_img, (1, 1))

    def add_score(self, x):
        global bar_score
        bar_score += x


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
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()

#turn = 0
while True:
    #turn += 1
    #if(turn == 51):
        #print "finish"
        #break
    main()

    keynum = 0
    stock = 0
    final = False
