#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os
import random
import math
from math import *

SCR_RECT = Rect(0, 0, 160, 160)
bar_score = 0
FINISH_SCORE = 100
perepisode = 5000
terminal_key = False
keynum = 0
stock = 0
times = 0
last_time = 0
final= False
angle = 360
# stepsfile = open("stepsdata/stepsdata.txt", "a")


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
    """
    Brick(3, 4)
    Hole(0, 0)
    """
    Brick(2, 5)
    Brick(4, 5)
    Brick(6, 5)
    Brick(8, 5)
    Hole(0, 0)
    Hole(1.5, 0)
    Hole(3.0, 0)
    Hole(4.5, 0)
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
    while len(holes) > 0 and keynum < perepisode and final == False:
        clock.tick(60)
        screen.fill((0, 0, 0))
        all.update()
        all.draw(screen)
        #score_board.draw(screen)  # スコアボードを描画
        pygame.display.update()
    terminal_key = True

#    stepsfile.write(str(keynum) + "\n")
#    print "The agent have moved %s steps" % keynum


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""
    def __init__(self, bricks, holes, score_board, angle, all):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle4.png")
        colorkey = self.image.get_at((0, 1))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self._image = self.image
        self.forward = 10
        self.back = 5
        self.angle = angle
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.alls = all
        self.score_board = score_board
        # self.x = SCR_RECT.bottom  # パドルは画面の一番下
        self.rect.center = (80,150)
        self.x, self.y = self.rect.center

    def rotate(self):
         center = self.rect.center
         self.image = pygame.transform.rotozoom(self._image, self.angle, 1.0)
         self.rect = self.image.get_rect(center=center)

    def update(self):
        self.image
        self._rect = Rect(self.rect)
        self._rect.center = self.x, self.y
        self.rotate()

        turn_speed = 30
        pygame.key.get_pressed()
        global keynum
        global times , final
        global bar_score, stock
        # 一回の移動量
        vx = vy = 5
        # すべてのブロックに対して
        # 一回の移動量
        vx = vy = 8

        for event in pygame.event.get():  # User did something
            if event.type == KEYDOWN:
                keynum += 1
                times += 1
                bar_score = 0
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
                        # print bar_score
                    # ボールが下から衝突
                    if brick.rect.top < oldrect.top < brick.rect.bottom < oldrect.bottom:
                        brick.rect.top -= vy
                        # print  bar_score
                    # bricks_collided2 = pygame.sprite.spritecollideany(brick, self.bricks,True)
                    # if bricks_collided2:
                        # for b in bricks_collided2:


                    if brick.rect.left < 0 and brick.rect.top < 0:
                        brick.rect.left += 16
                        brick.rect.top += 16
                    if brick.rect.left < 0 and brick.rect.bottom > 160:
                        brick.rect.left += 16
                        brick.rect.top -= 16
                    if brick.rect.right > 160 and brick.rect.top < 0:
                        brick.rect.left -= 16
                        brick.rect.top += 16
                    if brick.rect.right > 160 and brick.rect.bottom > 160:
                        brick.rect.left -= 16
                        brick.rect.top -= 16
                    # ボールが画面外に出たらリセット
                    if brick.rect.left < 0:
                        brick.rect.left = 0
                    if brick.rect.top < 0:
                        brick.rect.top = 0
                    if brick.rect.right > 160:
                        brick.rect.right = 160
                    if brick.rect.bottom > 160:
                        brick.rect.bottom = 160


                            # ボールが画面外に出たらリセット
                    # if brick.rect.left < 0 or brick.rect.left > 159 or brick.rect.top < 0 or brick.rect.top > 159:
                        # final = True
                        # bar_score = -1

                    bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, True)
                    if bricks_collided2:
                        pygame.sprite.Group.remove(self.bricks, brick)
                        pygame.sprite.Group.remove(self.alls, brick)
                        bar_score += 5

            self.rect.clamp_ip(SCR_RECT)

            if len(self.holes) == 0:
                bar_score += 10
            # print "hole=",len(self.holes)


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("brick3.png")
        colorkey=self.image.get_at((0,0))
        self.image.set_colorkey(colorkey,RLEACCEL)
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("hole3.png")
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

turn = 0
#s_turn = 0
while True:
#    s_turn += 1
#    if s_turn == 10:
#        print "finish"
#        stepsfile.write("finish\n")
#        s_turn = 0
#        turn += 1
    main()

    keynum = 0
    stock = 0
    final = False
