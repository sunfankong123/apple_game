#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os

SCR_RECT = Rect(0, 0, 320, 240)
bar_score = 0
# FINISH_SCORE = 100
perepisode = 200
terminal_key = False
keynum = 0
stock = 0
times = 0
last_time = 0
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
    Hole.containers = all, holes

    for x in range(1, 10, 2):  # 1列から10列まで
        for y in range(1, 5, 2):  # 1行から5行まで
            Brick(x, y)
    Hole(16, 1)
    Hole(18, 1)
    Hole(16, 3)
    Hole(18, 3)
    for x in range(6, 12, 2):  # 1列から10列まで
        for y in range(10, 13, 2):  # 1行から5行まで
            Hole(x,y)
    # スコアボードを作成
    score_board = ScoreBoard()

    # パドル
    paddle = Paddle(bricks, holes, score_board)

    clock = pygame.time.Clock()

    global perepisode, terminal_key, times, last_time
    terminal_key = False

    # current_time = int(times / 100000)
    # if current_time != last_time and perepisode > 120:
    #    last_time = current_time
    #    perepisode -= 10

    while len(holes) > 0 and keynum < perepisode:
        clock.tick(60)
        screen.fill((0, 0, 0))
        all.update()
        all.draw(screen)
        score_board.draw(screen)  # スコアボードを描画
        pygame.display.update()
        paddle.update()
    terminal_key = True

    # stepsfile.write(str(keynum) + "\n")
    # print "The agent have moved %s steps" % keynum


class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""
    def __init__(self, bricks, holes, score_board):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("paddle3.png")
        colorkey = self.image.get_at((0, 5))
        self.image.set_colorkey(colorkey, RLEACCEL)
        self.bricks = bricks  # ブロックグループへの参照
        self.holes = holes
        self.score_board = score_board
        self.rect.bottom = SCR_RECT.bottom  # パドルは画面の一番下

    def update(self):
        pygame.key.get_pressed()
        global keynum
        global times
        vx = vy = 20
        for event in pygame.event.get():  # User did something
            if event.type == KEYDOWN:
                keynum += 1
                times += 1
                bar_score = 0
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

            # ブロックを壊す
        # pygame.sprite.collide_rect
        global bar_score, stock
        # ボールと衝突したブロックリストを取得
        if stock <= 0:
            bricks_collided = pygame.sprite.spritecollide(self, self.bricks, True)

            if bricks_collided:  # 衝突ブロックがある場合
                if len(bricks_collided) == 1:
                    stock += 1
                    bar_score += 1
                elif len(bricks_collided) == 2:
                    stock += 2
                    bar_score += 2
            else:
                bar_score = -0.1
        elif stock == 1 or stock == 2 or stock == 3 or stock == 4:
            bricks_collided = pygame.sprite.spritecollide(self, self.bricks,True)
            if bricks_collided :
                if len(bricks_collided) == 1:
                    stock += 1
                    bar_score += 1
                elif len(bricks_collided) == 2:
                    stock += 2
                    bar_score += 2
            else:
                bricks_collided1 = pygame.sprite.spritecollide(self, self.holes, True)
                if bricks_collided1:
                    if len(bricks_collided1) == 1:
                        stock -= 1
                        bar_score += 1
                    elif len(bricks_collided1) == 2:
                        stock -= 2
                        bar_score += 2
                else:
                    bar_score -= 0.1
        elif stock > 4 :

            bricks_collided = pygame.sprite.spritecollide(self, self.bricks, False)
            if bricks_collided:  # 衝突ブロックがある場合
                oldrect = self.rect
                for brick in bricks_collided:  # 各衝突ブロックに対して
                    # ボールが左から衝突
                    if oldrect.left < brick.rect.left < oldrect.right < brick.rect.right:
                        self.rect.right = brick.rect.left
                    # ボールが右から衝突
                    if brick.rect.left < oldrect.left < brick.rect.right < oldrect.right:
                        self.rect.left = brick.rect.right
                    # ボールが上から衝突
                    if oldrect.top < brick.rect.top < oldrect.bottom < brick.rect.bottom:
                        self.rect.bottom = brick.rect.top
                    # ボールが下から衝突
                    if  brick.rect.top < oldrect.top < brick.rect.bottom < oldrect.bottom:
                        self.rect.top = brick.rect.bottom

            bricks_collided = pygame.sprite.spritecollide(self, self.holes, True)

            if bricks_collided:
                if len(bricks_collided) == 1:
                    stock -= 1
                    bar_score -= 1
                elif len(bricks_collided) == 2:
                    stock -= 2
                    bar_score -= 2
            else:
                bar_score -= 0.1
        self.rect.clamp_ip(SCR_RECT)
        # print bar_score
class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("brick2.png")
        colorkey=self.image.get_at((0,0))
        self.image.set_colorkey(colorkey,RLEACCEL)
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image("hole.png")
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height


class ScoreBoard():
    """スコアボード"""
    def __init__(self):
        self.sysfont = pygame.font.SysFont(None, 17)
        # self.score = 0

    def draw(self, screen):
        score_img = self.sysfont.render("SCORE:" + str(bar_score), True, (255, 255, 255))
        screen.blit(score_img, (1, 1))

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
