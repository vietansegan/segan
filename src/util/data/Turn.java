/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.data;

/**
 *
 * @author vietan
 */
public class Turn implements Comparable<Turn> {

    private int index; // the turn order in the conversation
    private String speaker;
    private String text;
    private int[] tokens;

    public Turn(int index, String speaker, String text) {
        this.index = index;
        this.speaker = speaker;
        this.text = text;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public String getSpeaker() {
        return speaker;
    }

    public void setSpeaker(String speaker) {
        this.speaker = speaker;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public int[] getTokens() {
        return tokens;
    }

    public void setTokens(int[] tokens) {
        this.tokens = tokens;
    }

    @Override
    public int hashCode() {
        return Integer.toString(index).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if ((obj == null) || (this.getClass() != obj.getClass())) {
            return false;
        }
        Turn t = (Turn) (obj);

        return (this.index == t.index);
    }

    @Override
    public int compareTo(Turn t) {
        return Integer.valueOf(index).compareTo(Integer.valueOf(t.index));
    }

    @Override
    public String toString() {
        return this.index + "\t" + this.speaker + "\t" + this.text;
    }
}
