

#biblioteca(s) importada(s )
import pandas as pd


class Pre_processing ():

    def import_files(self):
        self.df1 = pd.read_csv ( "Youtube01-Psy.csv" )
        self.df2 = pd.read_csv ( "Youtube02-KatyPerry.csv" )
        self.df3 = pd.read_csv ( "Youtube05-Shakira.csv" )
        self.df4 = pd.read_csv ( "Youtube04-Eminem.csv" )
        self.df5 = pd.read_csv ( "Youtube03-LMFAO.csv" )

    def concatenate_files(self):
        list_files = [
            [ "PSY" , self.df1 ] ,
            [ "KATY PERRY" , self.df2 ] ,
            [ "SHAKIRA" , self.df3 ] ,
            [ "EMINEM" , self.df4 ] ,
            [ "LMFAO" , self.df5 ] ,
        ]

        self.insert_artist_column ( list_files )

        new_csv = pd.concat ( [ file[ 1 ] for file in list_files ] )
        n_list = self.delete_no_useful_data ( new_csv )
        new_csv.to_csv ( "merged.csv" )


    def fill_missing_data(self , list_files):

        list_files[ "DATE" ].fillna ( method="bfill" , inplace=True )


    def delete_no_useful_data(self , database):
        database.drop ( 'COMMENT_ID' , axis=1 , inplace=True )
        return database

    def insert_artist_column(self , list_files):

        list_files = list ( list_files )
        if (len ( list_files ) == 0):
            return
        else:

            list_files[ -1 ][ 1 ][ "ARTIST" ] = pd.np.nan
            self.fill_missing_data ( list_files[ -1 ][ 1 ] )
            list_files[ -1 ][ 1 ].fillna ( list_files[ -1 ][ 0 ] , inplace=True )
            list_files.pop ( -1 )
            self.insert_artist_column ( list_files )

p = Pre_processing()
p.import_files()
p.concatenate_files()