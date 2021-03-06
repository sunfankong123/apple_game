#!/usr/bin/env python
# coding: utf-8
import pygame
from pygame.locals import *
import os
import random
import math
from math import *
screen_x = 192
screen_y = 192
SCR_RECT = Rect(0, 0, screen_x, screen_y)
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
fileinputdata = open("inputdata.txt", "w")


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
    for x in range(1, 8, 1):
        for y in range(1, x, 1):
            Hole(x, 0, 1)
    for x in range(6, 13, 1):
        for y in range(1, x - 5, 1):
            Brick(x, 6, 1)
    """
    Hole(1, 0, 2)
    Hole(1, 0, 2)
    Hole(2, 0, 2)
    Hole(2, 0, 2)
    Hole(3, 0, 2)
    Hole(3, 0, 2)
    Hole(4, 0, 2)
    Hole(4, 0, 2)

    Hole(1, 1, 1)
    Hole(2, 1, 1)
    Hole(3, 1, 1)
    Hole(4, 1, 1)


    Brick(3, 6, 2)
    Brick(3, 6, 2)
    Brick(2, 6, 1)
    Brick(4, 6, 1)
    Brick(3, 5, 1)
    Brick(3, 7, 1)

    Brick(8, 6, 2)
    Brick(8, 6, 2)
    Brick(7, 6, 1)
    Brick(9, 6, 1)
    Brick(8, 5, 1)
    Brick(8, 7, 1)
    # パドル
    paddle = Paddle(bricks, holes, angle, all)

    clock = pygame.time.Clock()

    global perepisode, terminal_key, times, last_time, bar_score
    terminal_key = False

    # current_time = int(times / 100000)
    # if current_time != last_time and perepisode > 120:
    #    last_time = current_time
    #    perepisode -= 10
    while len(holes) > 0 and keynum < perepisode and final == False:
        clock.tick(60)
        screen.fill((128, 128, 128))
        all.update()
        all.draw(screen)
        pygame.display.update()
    terminal_key = True
#    stepsfile = open("stepsdata_networks_termina/stepsdata3600000.txt", "a")
#    stepsfile.write(str(keynum) + "\n")
#    print "The agent have moved %s steps" % keynum

class Paddle(pygame.sprite.Sprite):
    """ボールを打つパドル"""
    def __init__(self, bricks, holes, angle, all):
        # containersはmain()でセットされる
        pygame.sprite.Sprite.__init__(self, self.containers)
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
        self.rect.center = (88,168)
        self.x, self.y = self.rect.center
    def rotate(self):
         center = self.rect.center
         self.image = pygame.transform.rotate(self._image, self.angle)
         self.rect = self.image.get_rect(center=center)

    def update(self):
        self._rect = Rect(self.rect)
        self._rect.center = self.x, self.y
        self.rotate()
        # turn_speed = 90
        # pygame.key.get_pressed()
        global keynum
        global times , final
        global bar_score, stock
        num = 0
        num2 = 0
        a = []
        # すべてのブロックに対して
        # 一回の移動量
        vx = vy = 16
        keyup = False
        keydown = False
        keyright = False
        keyleft = False
        for event in pygame.event.get():  # User did something
            if event.type == KEYDOWN:
                bar_score = 0
                keynum += 1
                times += 1
                # 重機の操縦(初代+旋回simple)
                if event.key == K_UP:
                    keyup = True
                    fileinputdata.write("1\n")
                    self.rect.move_ip(0, -vy)
                    self.angle = 0
                elif event.key == K_DOWN:
                    keydown = True
                    fileinputdata.write("0\n")
                    self.rect.move_ip(0, vy)
                    self.angle = 180
                elif event.key == K_RIGHT:
                    keyright = True
                    fileinputdata.write("3\n")
                    self.rect.move_ip(vx, 0)
                    self.angle = 270
                elif event.key == K_LEFT:
                    keyleft = True
                    fileinputdata.write("2\n")
                    self.rect.move_ip(-vx, 0)
                    self.angle = 90

            # 重機と衝突したブロックリストを取得
            bricks_collided = pygame.sprite.spritecollide(self, self.bricks, False)
            if bricks_collided:  # 衝突ブロックがある場合

                for brick in bricks_collided:  # 各衝突ブロックに対して
                    oldrect = brick.rect
                    if keyup == True:
                        brick.rect.top -= 16
                        bar_score += 1
                    if keydown == True:
                        brick.rect.top += 16
                        bar_score -= 1
                    if keyright == True:
                        brick.rect.left += 16
                    if keyleft == True:
                        brick.rect.left -= 16

                    a = pygame.sprite.spritecollide(brick, self.bricks, False)

                    # 四辺の判定
                    if brick.rect.left < 0:
                        brick.rect.left = 16
                        bar_score -= 1
                    if brick.rect.top < 0:
                        brick.rect.top = 16
                        bar_score -= 1
                    if brick.rect.right > screen_x:
                        brick.rect.right = screen_x - 1
                        bar_score -= 1
                    if brick.rect.bottom > screen_y:
                        brick.rect.bottom = screen_y -1
                        bar_score -= 1

                    # brickとholeの衝突判定
                    bricks_collided2 = pygame.sprite.spritecollide(brick, self.holes, False)
                    if bricks_collided2:
                        pygame.sprite.Group.remove(self.bricks, brick)
                        pygame.sprite.Group.remove(self.alls, brick)
                        # 衝突しているbrickとholeをgroupから削除
                        for hole in bricks_collided2:
                            pygame.sprite.Group.remove(self.holes, hole)
                            pygame.sprite.Group.remove(self.alls, hole)
                            bar_score += 5
                            break

            # brick同士の衝突(重ねられなくする処理)
            if len(a) > 4:
                for brick in bricks_collided:
                    num2 += 1
                    if keyup == True:
                        brick.rect.top += 16
                        bar_score -= 1
                    if keydown == True:
                        brick.rect.top -= 16
                    if keyright == True:
                        brick.rect.left -= 16
                    if keyleft == True:
                        brick.rect.left += 16
                    if len(bricks_collided) - num2 == 3:
                        break
            # brick同士の衝突(取り残される処理)
            elif len(bricks_collided) > 3:
                for brick in bricks_collided:
                    num += 1
                    if keyup == True:
                        brick.rect.top += 16
                    if keydown == True:
                        brick.rect.top -= 16
                    if keyright == True:
                        brick.rect.left -= 16
                    if keyleft == True:
                        brick.rect.left += 16
                    if len(bricks_collided) - num == 3:
                        break

            # 土砂の高さをdepthに変換
            for brick in self.bricks:
                old_x = brick.rect.left / 16
                old_y = brick.rect.top / 16
                pygame.sprite.Group.remove(self.bricks, brick)
                pygame.sprite.Group.remove(self.alls, brick)
                a = pygame.sprite.spritecollide(brick, self.bricks, False)
                Brick(old_x, old_y, len(a))

            # 穴の深さをdepthに変換
            for hole in self.holes:
                old_x = hole.rect.left / 32
                old_y = hole.rect.top / 16
                pygame.sprite.Group.remove(self.holes, hole)
                pygame.sprite.Group.remove(self.alls, hole)
                b = pygame.sprite.spritecollide(hole, self.holes, False)
                Hole(old_x, old_y, len(b))

            self.rect.clamp_ip(SCR_RECT)

            # 穴を全部埋めたら報酬大
            if len(self.holes) == 0:
                bar_score += 10
            # print "socore=",bar_score
            # print "hole=",len(self.holes)


class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
        global bar_score, terminal_key
        self.level = level
        self.depth = 128 + (self.level + 1) * 31
        if self.depth > 255:
            self.depth = 255
            bar_score -= 10
            terminal_key = True
        self.image = pygame.Surface((16, 16), 0, 32)
        self.rect = self.image.fill((self.depth, self.depth, self.depth))
        # ブロックの位置を更新
        self.rect.left = SCR_RECT.left + x * self.rect.width
        self.rect.top = SCR_RECT.top + y * self.rect.height

class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y, level):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.level = level
        self.depth = 128 - (self.level + 1) * 31
        self.image = pygame.Surface((32, 16), 0, 32)
        self.rect = self.image.fill((self.depth, self.depth, self.depth))
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