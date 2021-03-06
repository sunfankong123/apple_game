#!/usr/bin/env python
#coding: utf-8
import pygame
from pygame.locals import *
import math
import os
import sys
import random
import numpy

SCR_RECT = Rect(0, 0, 640, 480)
bar_score = 0
INVALID_ACTIONS = 30
FINISH_SCORE = 400
key_num = 0


def main():

    pygame.init()
    screen = pygame.display.set_mode(SCR_RECT.size,0,32)
    # pygame.display.set_caption(u"Breakout 06 スコアの表示")

    # スプライトグループを作成して登録
    all = pygame.sprite.RenderUpdates()  # 描画用グループ
    bricks = pygame.sprite.Group()       # 衝突判定用グループ
    Paddle.containers = all
    Brick.containers = all, bricks


    # ブロックを作成
    # 自動的にbricksグループに追加される

    Brick(1,23)

    """
    Brick(1, 24)
    Brick(2, 24)
    Brick(3, 24)
    """

    a=random.randint(0,5)
    b=random.randint(0,14)
    for x in range(0,6,2):
        for y in range(0,4,2):
            Brick(a+x, b+y)

    a = random.randint(0, 5)
    b = random.randint(19, 24)
    for x in range(0, 5, 2):
        for y in range(0, 4, 2):
            Brick(a + x, b + y)

    a = random.randint(14,34 )
    b = random.randint(0, 14)
    for x in range(0, 5, 2):
        for y in range(0, 4, 2):
            Brick(a + x, b + y)

    a = random.randint(14, 34)
    b = random.randint(19, 24)
    for x in range(0, 6, 2):
        for y in range(0, 6, 2):
            Brick(a + x, b + y)

    for i in range (1,20):
        Brick(random.randint(0,39), random.randint(0,29))



    # スコアボードを作成
    score_board = ScoreBoard()

    # パドル
    paddle = Paddle(bricks, score_board)

    clock = pygame.time.Clock()

    #while len():
    #while bar_score < FINISH_SCORE:
    while len(bricks) > 1 and key_num < 500:
        clock.tick(60)
        screen.fill((0,0,0))
        all.update()
        all.draw(screen)
        score_board.draw(screen)  # スコアボードを描画
        pygame.display.update()
        paddle.update()





class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""
    def __init__(self, bricks, score_board):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle2.png")
        colorkey = self.image.get_at((0, 5))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self.bricks = bricks  # ブロックグループへの参照
        self.score_board = score_board
        self.rect.left=random.randint(0,630) # パドルは画面の一番下
        self.rect.top=random.randint(0,470)

    def update(self):
        #self.rect.centerx = pygame.mouse.get_pos()[0]  # パドルの中央のX座標=マウスのX座標

        pressed_keys = pygame.key.get_pressed()

        vx = vy = 20

        global key_num
        done = False
        push=0
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this
            if event.type == KEYDOWN:
                key_num += 1
                if event.key == K_UP:
                    self.rect.move_ip(0, -vy)
                elif event.key == K_DOWN:
                    self.rect.move_ip(0, vy)
                elif event.key == K_RIGHT:
                    self.rect.move_ip(vx, 0)
                elif event.key == K_LEFT:
                    self.rect.move_ip(-vx, 0)
            elif event.type == KEYUP:
                if event.key == K_UP:
                    self.rect.move_ip(0, 0)
                elif event.key == K_DOWN:
                    self.rect.move_ip(0, 0)
                elif event.key == K_RIGHT:
                    self.rect.move_ip(0, 0)
                elif event.key == K_LEFT:
                    self.rect.move_ip(0, 0)
        '''
        # 押されているキーに応じて画像を移動
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-vx, 0)
        if pressed_keys[K_RIGHT]:
            self.rect.move_ip(vx, 0)
        if pressed_keys[K_UP]:
            self.rect.move_ip(0, -vy)
        if pressed_keys[K_DOWN]:
            self.rect.move_ip(0, vy)
            '''
            # ブロックを壊す
        global bar_score
        # ボールと衝突したブロックリストを取得
        bricks_collided = pygame.sprite.spritecollide(self, self.bricks, True)

        if bricks_collided:  # 衝突ブロックがある場合

                # 点数を追加
            bar_score = 1
        else:
            bar_score = 0

        self.rect.clamp_ip(SCR_RECT)


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("brick2.png")
        colorkey=self.image.get_at((0,0))
        self.image.set_colorkey(colorkey,RLEACCEL)
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class ScoreBoard():
    """スコアボード"""
    def __init__(self):
        self.sysfont = pygame.font.SysFont(None, 20)
        # self.score = 0

    def draw(self, screen):
        score_img = self.sysfont.render("SCORE:" + str(bar_score), True, (255, 255, 250))
        screen.blit(score_img, (1, 1))
    """
    def draw(self, screen):
        img = self.sysfont.render("SCORE:" + str(self.score), True, (255, 255, 250))
        screen.blit(img, (self.x, self.y))
    """
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
    bar_score = 0
    key_num = 0




